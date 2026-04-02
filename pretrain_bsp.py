# pretrain_bsp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.io as io
import numpy as np
import os
from timesformer_bsp import TimeSformer_BSP
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# Dataset
# ===============================
class VideoDataset(Dataset):
    """
    Read video paths from a txt file, format:
    /path/to/video1.mp4 0
    /path/to/video2.mp4 1
    """
    def __init__(self, list_file, transform=None):
        self.data_info = self.get_video_info(list_file)
        self.transform = transform
        print(f"Dataset loaded: {len(self.data_info)} samples")

    def __getitem__(self, index):
        video_path, label = self.data_info[index]
        try:
            # Read video frames (T, H, W, C)
            video, _, _ = io.read_video(video_path, pts_unit="sec")

            # Check for empty video (T == 0)
            if video.shape[0] == 0:
                print(f"[Warning] Empty video detected: {video_path}")
                raise ValueError("Empty video")

            video = video.float() / 255.0  # normalize to [0,1]

            # Ensure 3 channels
            if video.shape[-1] == 1:
                video = video.repeat(1, 1, 1, 3)
            elif video.shape[-1] > 3:
                video = video[:, :, :, :3]

            # Apply transform if provided
            if self.transform:
                video = self.transform(video)

            return video, label

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return torch.zeros((32, 224, 224, 3)), 0

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_video_info(list_file):
        data_info = []
        with open(list_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                # Case 1: contains comma, e.g., path,label
                if ',' in line:
                    path, label = line.split(',')
                    label = int(label)

                # Case 2: contains space, e.g., path label
                elif ' ' in line:
                    parts = line.split()
                    path = parts[0]
                    label = int(parts[1]) if len(parts) > 1 else 0

                # Case 3: only path without label
                else:
                    path = line
                    label = 0

                data_info.append((path, label))

        return data_info


# ===============================
# Collate function
# ===============================
def custom_collate_fn(batch):
    videos, labels = zip(*batch)

    # Convert to tensors
    videos = [torch.as_tensor(v, dtype=torch.float32) for v in videos]

    # Find max temporal length and spatial size
    max_t = max(v.shape[0] for v in videos)
    max_h = max(v.shape[1] for v in videos)
    max_w = max(v.shape[2] for v in videos)

    # Or fix to a target size (recommended)
    target_h, target_w = 224, 224

    padded_videos = []
    for v in videos:
        T, H, W, C = v.shape

        # Handle temporal dimension
        if T < max_t:
            pad_t = max_t - T
            v = torch.cat([v, v[-1:].repeat(pad_t, 1, 1, 1)], dim=0)
        elif T > max_t:
            v = v[:max_t]

        # Handle spatial dimensions - resize to fixed size
        if H != target_h or W != target_w:
            # (T, H, W, C) -> (T, C, H, W)
            v_permuted = v.permute(0, 3, 1, 2)

            # Resize using interpolation
            v_resized = F.interpolate(
                v_permuted,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

            # (T, C, H, W) -> (T, H, W, C)
            v = v_resized.permute(0, 2, 3, 1)

        padded_videos.append(v)

    videos = torch.stack(padded_videos)  # (B, T, H, W, C)
    labels = torch.tensor(labels)

    return videos, labels


# ===============================
# Pretrain loop
# ===============================
def pretrain_loop(model, dataloader, optimizer, device, epochs=10, save_path=None, log_path=None):
    model.train()

    if hasattr(model, "switch_mode"):
        model.switch_mode("pretrain")

    mse = nn.MSELoss()

    # If log_path is specified, write header (only when file does not exist)
    if log_path and not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])

    for epoch in range(epochs):
        total_loss = 0.0

        for i, (videos, _) in enumerate(dataloader):
            # (B, T, H, W, C) -> (B, C, T, H, W)
            videos = videos.permute(0, 4, 1, 2, 3).to(device)

            # Ensure fixed 32 frames
            B, C, T, H, W = videos.shape
            if T < 32:
                repeat_times = 32 // T + 1
                videos = videos.repeat(1, 1, repeat_times, 1, 1)[:, :, :32]
            elif T > 32:
                videos = videos[:, :, :32]

            # Update shape
            B, C, T, H, W = videos.shape

            # Resize spatial resolution to 224x224 (per frame)
            if H != 224 or W != 224:
                # (B, C, T, H, W) -> (B*T, C, H, W)
                videos = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

                videos = F.interpolate(videos, size=(224, 224), mode="bilinear")

                # (B*T, C, 224, 224) -> (B, C, T, 224, 224)
                videos = videos.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)

            optimizer.zero_grad()

            # Pretraining: reconstruct video
            reconstructed = model(videos)
            loss = mse(reconstructed, videos)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"[Epoch {epoch} Batch {i}] Loss: {loss.item():.6f}")

        avg_loss = total_loss / (i + 1)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.6f}")

        # Save to CSV log
        if log_path:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss])

        # Save checkpoint
        if save_path and epoch % 10 == 0:
            ckpt = save_path.replace(".pth", f"_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt)
            print(f"Checkpoint saved to {ckpt}")


# ===============================
# Train entry
# ===============================
def train_pretrain(list_file, batch_size=2, epochs=10, store_name="time_pretrain"):
    os.makedirs(store_name, exist_ok=True)

    dataset = VideoDataset(list_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    model = TimeSformer_BSP(
        dim=512,
        image_size=224,
        num_classes=2,
        num_frames=32,
        patch_size=16,
        mode="pretrain"
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    save_path = f"./{store_name}/bsp_pretrained.pth"
    log_path = f"./{store_name}/pretrain_log.csv"

    pretrain_loop(
        model,
        dataloader,
        optimizer,
        device,
        epochs=epochs,
        save_path=save_path,
        log_path=log_path
    )


if __name__ == "__main__":
    # Training data txt file
    list_file = "xxxxxxx"
    train_pretrain(list_file, batch_size=2, epochs=100, store_name="xxxxxxx")