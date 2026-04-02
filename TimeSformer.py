import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from timesformer_pytorch.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding


# ======================
# helpers
# ======================

def exists(val):
    return val is not None


# ======================
# modules
# ======================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


def shift(t, amt):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))


class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f=f)

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim=-1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(
            map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1)))
        )
        x = torch.cat((*shifted_chunks, *rest), dim=-1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim=1)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


def attn(q, k, v, mask=None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self, x, einops_from, einops_to,
        mask=None, cls_mask=None, rot_emb=None, **einops_dims
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale

        # cls token
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, :1], t[:, 1:]), (q, k, v)
        )
        cls_out = attn(cls_q, k, v, mask=cls_mask)

        # rearrange
        q_, k_, v_ = map(
            lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims),
            (q_, k_, v_)
        )

        # rotary embedding
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(
            lambda t: repeat(t, 'b () d -> (b r) () d', r=r),
            (cls_k, cls_v)
        )
        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        out = attn(q_, k_, v_, mask=mask)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        out = torch.cat((cls_out, out), dim=1)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# ======================
# TimeSformer
# ======================

class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes=None,
        image_size=224,
        patch_size=16,
        channels=3,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.,
        rotary_emb=True,
        shift_tokens=False
    ):
        super().__init__()
        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            time_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            spatial_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(
                    lambda t: PreTokenShift(num_frames, t),
                    (time_attn, spatial_attn, ff)
                )

            time_attn, spatial_attn, ff = map(
                lambda t: PreNorm(dim, t),
                (time_attn, spatial_attn, ff)
            )
            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.has_classifier = num_classes is not None
        if self.has_classifier:
            self.to_out = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, video, mask=None, return_features=False, return_cls=False):
        video = video.permute(0, 2, 1, 3, 4)
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        hp, wp = (h // p), (w // p)
        n = hp * wp

        video = rearrange(
            video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p
        )
        tokens = self.to_patch_embedding(video)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, tokens), dim=1)

        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device=device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device=device)
            image_pos_emb = self.image_rot_emb(hp, wp, device=device)

        frame_mask = None
        cls_attn_mask = None
        if exists(mask):
            mask_with_cls = F.pad(mask, (1, 0), value=True)
            frame_mask = repeat(mask_with_cls, 'b f -> (b h n) () f', n=n, h=self.heads)
            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n=n, h=self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value=True)

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n,
                          mask=frame_mask, cls_mask=cls_attn_mask,
                          rot_emb=frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f,
                             cls_mask=cls_attn_mask,
                             rot_emb=image_pos_emb) + x
            x = ff(x) + x

        if return_features:
            return x
        else:
            cls_token = x[:, 0]
            if return_cls:
                return cls_token
            return self.to_out(cls_token)

    def get_patch_features(self, video):
        features = self.forward(video, return_features=True)
        return features[:, 1:]


# ======================
# TwoStreamTimeSformer
# ======================

class TwoStreamTimeSformer(nn.Module):
    def __init__(self, dim=512, image_size=224, num_classes=2, patch_size=16, num_frames=32):
        super(TwoStreamTimeSformer, self).__init__()

        self.stream1 = TimeSformer(
            dim=dim,
            image_size=image_size,
            num_classes=None,
            num_frames=num_frames,
            patch_size=patch_size,
            channels=3
        )

        self.stream2 = TimeSformer(
            dim=dim,
            image_size=image_size,
            num_classes=None,
            num_frames=num_frames,
            patch_size=patch_size,
            channels=3
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_classes)
        )

        self.cross_attn = CrossAttention(dim=dim)

    def forward(self, x1, x2):
        feat1 = self.stream1(x1, return_cls=True)
        feat2 = self.stream2(x2, return_cls=True)

        # Cross-attention for modality interaction
        feat1_new = self.cross_attn(feat1, feat2)
        feat2_new = self.cross_attn(feat2, feat1)

        # Feature fusion
        combined = torch.cat([feat1_new, feat2_new], dim=-1)
        return self.fusion(combined)

    def forward_single_stream(self, x, stream_id=1):
        if stream_id == 1:
            return self.stream1.get_patch_features(x)
        else:
            return self.stream2.get_patch_features(x)


# Cross-Attention module
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat1, feat2):
        # Inputs feat1, feat2: (B, D)
        B, D = feat1.shape
        H = self.num_heads
        Hd = self.head_dim

        # Expand to multi-head format
        q = self.q_proj(feat1).view(B, H, Hd).unsqueeze(2)
        k = self.k_proj(feat2).view(B, H, Hd).unsqueeze(2)
        v = self.v_proj(feat2).view(B, H, Hd).unsqueeze(2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Context aggregation
        context = torch.matmul(attn_probs, v)
        context = context.squeeze(2).reshape(B, D)

        out = self.out_proj(context)
        out = self.norm(out + feat1)

        return out


# Lightweight Translution + CrossAttention module
class LightTransCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3, dropout=0.1):
        super(LightTransCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Local structure enhancement (depthwise convolution + learnable positional bias)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.rel_pos = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.rel_pos, std=0.02)

    def forward(self, feat1, feat2):
        """
        feat1: modality 1 features (B, D)
        feat2: modality 2 features (B, D)
        """
        B, D = feat1.shape
        H = self.num_heads
        Hd = self.head_dim

        # Local dependency enhancement
        f1_enhanced = self.local_conv(feat1.unsqueeze(-1)).squeeze(-1)
        f2_enhanced = self.local_conv(feat2.unsqueeze(-1)).squeeze(-1)

        # Cross-attention
        q = self.q_proj(f1_enhanced).view(B, H, Hd)
        k = self.k_proj(f2_enhanced).view(B, H, Hd)
        v = self.v_proj(f2_enhanced).view(B, H, Hd)

        attn = torch.matmul(q.unsqueeze(2), k.unsqueeze(2).transpose(-2, -1)) / (Hd ** 0.5)
        attn = attn + self.rel_pos
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v.unsqueeze(2)).squeeze(2)
        context = context.reshape(B, D)

        out = self.out_proj(context)
        out = self.norm(out + feat1)
        return out