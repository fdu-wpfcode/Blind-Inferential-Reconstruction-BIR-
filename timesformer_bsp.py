# timesformer_bsp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from TimeSformer import TwoStreamTimeSformer
from TimeSformer import TimeSformer


class BlindSpotConv3D(nn.Module):
    """
    3D convolution where the kernel center is zeroed (blind-spot).
    This does NOT modify the original conv weights in-place.
    Input/Output: (B, C, T, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, bias=bias)
        # prepare mask buffer with same shape as weight
        with torch.no_grad():
            mask = torch.ones_like(self.conv.weight.data)
            k = kernel_size
            center = k // 2
            # set central element to 0 for all input/output channels
            mask[:, :, center, center, center] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        # masked convolution via functional API to avoid in-place modification
        masked_weight = self.conv.weight * self.mask
        return F.conv3d(x, masked_weight, bias=self.conv.bias,
                        stride=self.conv.stride,
                        padding=self.conv.padding,
                        dilation=self.conv.dilation,
                        groups=self.conv.groups)


class BlindSpotConv3D_mask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, bias=False, cross_mask=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

        # generate blind-spot mask
        with torch.no_grad():
            mask = torch.ones_like(self.conv.weight.data)
            kT, kH, kW = self.conv.kernel_size if isinstance(self.conv.kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
            cT, cH, cW = kT // 2, kH // 2, kW // 2

            if cross_mask:
                # cross-shaped masking: mask the three central axes
                mask[:, :, cT, :, :] = 0.0   # temporal axis slice
                mask[:, :, :, cH, :] = 0.0   # height axis slice
                mask[:, :, :, :, cW] = 0.0   # width axis slice
            else:
                # only mask the center point
                mask[:, :, cT, cH, cW] = 0.0

        self.register_buffer("mask", mask)

    def forward(self, x):
        masked_weight = self.conv.weight * self.mask
        return F.conv3d(x, masked_weight, bias=self.conv.bias,
                        stride=self.conv.stride,
                        padding=self.conv.padding,
                        dilation=self.conv.dilation,
                        groups=self.conv.groups)


#
# Helper: patchify/unpatchify for video
#
def video_to_patches(video, patch_size):
    # video: (B, C, T, H, W)
    B, C, T, H, W = video.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, "H/W must be divisible by patch_size"
    nh = H // p
    nw = W // p

    # -> (B, T, nh*nw, patch_dim)
    x = rearrange(video, 'b c t (nh p) (nw p) -> b t (nh nw) (p p c)', p=p)

    # flatten to (B, T*nh*nw, patch_dim)
    patches = rearrange(x, 'b t n pc -> b (t n) pc')
    return patches  # (B, num_tokens, patch_dim)


class TimeSformer_BSP(nn.Module):
    """
    Unified TimeSformer model supporting:
    - Single-stream pretraining (reconstruction task)
    - Two-stream classification task
    """

    def __init__(self, *, dim=512, image_size=224, num_classes=2, num_frames=32, patch_size=16, mode="pretrain"):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_classes = num_classes

        # Blind-spot convolution frontend
        self.bsp_frontend = BlindSpotConv3D_mask(
            in_channels=3,
            out_channels=3,
            kernel_size=5,
            padding=2,
            bias=False
        )

        if mode == "pretrain":
            # Pretraining mode: single-stream TimeSformer (no classifier head)
            self.backbone = TimeSformer(
                dim=dim,
                image_size=image_size,
                num_classes=None,
                num_frames=num_frames,
                patch_size=patch_size,
                channels=3
            )

            # Reconstruction projection head
            patch_dim = 3 * patch_size * patch_size
            self.reconstruction_proj = nn.Linear(dim, patch_dim)

        else:
            # Finetuning mode: two-stream TimeSformer (with classifier head)
            self.backbone = TwoStreamTimeSformer(
                dim=dim,
                image_size=image_size,
                num_classes=num_classes,
                num_frames=num_frames,
                patch_size=patch_size
            )

    def forward(self, x1, x2=None):

        if self.mode == "pretrain":
            # Single-stream pretraining: reconstruction task

            # Apply blind-spot convolution
            bsp_x1 = self.bsp_frontend(x1)
            if x2 is not None:
                bsp_x2 = self.bsp_frontend(x2)

            patch_features = self.backbone.get_patch_features(bsp_x1)

            # Project back to pixel space
            B, N, D = patch_features.shape
            patch_dim = 3 * self.patch_size * self.patch_size
            patches_pred = self.reconstruction_proj(patch_features)

            # Reconstruct video
            return self._reconstruct_video(patches_pred, B)

        else:
            # Finetuning mode (classification)
            bsp_x1 = x1
            bsp_x2 = x2
            return self.backbone(bsp_x1, bsp_x2)

    def _reconstruct_video(self, patches_pred, batch_size):
        """Reconstruct video from patches"""
        f = self.num_frames
        p = self.patch_size
        nh = self.image_size // p
        nw = nh
        total_patches = f * nh * nw

        # Ensure patch count matches
        if patches_pred.shape[1] != total_patches:
            # Truncate or pad if mismatch
            patches_pred = patches_pred[:, :total_patches]

        # reshape to video format
        patches_reshaped = patches_pred.view(batch_size, f, nh, nw, p, p, 3)
        video_recon = rearrange(
            patches_reshaped,
            'b f nh nw ph pw c -> b c f (nh ph) (nw pw)'
        ).contiguous()

        return video_recon

    def switch_mode(self, new_mode, num_classes=None):
        """Dynamically switch model mode"""
        if new_mode == self.mode:
            return

        self.mode = new_mode

        if new_mode == "pretrain":
            # Switch to pretraining mode
            self.backbone = TimeSformer(
                dim=512,
                image_size=self.image_size,
                num_classes=None,
                num_frames=self.num_frames,
                patch_size=self.patch_size,
                channels=3
            )

            patch_dim = 3 * self.patch_size * self.patch_size
            self.reconstruction_proj = nn.Linear(512, patch_dim)

        else:
            # Switch to finetuning mode
            num_classes = num_classes or self.num_classes
            self.backbone = TwoStreamTimeSformer(
                dim=512,
                image_size=self.image_size,
                num_classes=num_classes,
                num_frames=self.num_frames,
                patch_size=self.patch_size
            )