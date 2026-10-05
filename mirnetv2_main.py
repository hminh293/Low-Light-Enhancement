import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataset import get_loader
from models.mirnetv2_model import MIRNetV2


def main():
    # -----------------------------
    # Config
    # -----------------------------
    root_dir = "./dataset"
    batch_size = 8
    num_workers = 2
    epochs = 10
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Dataset
    # -----------------------------
    train_loader = get_loader(root_dir, "train", batch_size, num_workers)
    val_loader = get_loader(root_dir, "val", batch_size, num_workers)

    # -----------------------------
    # Model
    # -----------------------------
    model = MIRNetV2().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Checkpoint dir
    # -----------------------------
    ckpt_dir = "./checkpoints/Mirnetv2"
    os.makedirs(ckpt_dir, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train")

        for low_img, light_img in train_bar:
            low_img = low_img.to(device)
            light_img = light_img.to(device)

            optimizer.zero_grad()
            pred = model(low_img)
            loss = criterion(pred, light_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Train Loss = {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(
            ckpt_dir, f"mirnetv2_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)

    print("Training completed!")


if __name__ == "__main__":
    main()
