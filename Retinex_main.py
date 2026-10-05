import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import gc

from utils.dataset import get_loader
from models.Retinex import RetinexNet


########################################
# Loss functions
########################################
def illumination_smoothness_loss(L):
    dx = torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1])
    dy = torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :])
    return dx.mean() + dy.mean()


########################################
# Main
########################################
if __name__ == "__main__":

    root_dir = "./dataset"
    batch_size = 16
    num_workers = 4
    epochs = 10

    train_loader = get_loader(root_dir, "train", batch_size, num_workers)
    val_loader = get_loader(root_dir, "val", batch_size, num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RetinexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    l1_loss = nn.L1Loss()

    checkpoint_dir = "./checkpoints/Retinex"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    try:
        last_ckpt = sorted(os.listdir(checkpoint_dir))[-1]
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, last_ckpt)))
        start_epoch = int(last_ckpt.split("_")[-1].split(".")[0])
        print(f"Resume from epoch {start_epoch}")
    except:
        print("Train Retinex-Net from scratch")

    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for low_img, high_img in train_bar:
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            optimizer.zero_grad()

            enhanced, R, L, L_hat = model(low_img)

            loss_recon = l1_loss(enhanced, high_img)
            loss_reflect = l1_loss(R, high_img / (L + 1e-6))
            loss_smooth = illumination_smoothness_loss(L_hat)

            loss = loss_recon + 0.1 * loss_reflect + 0.01 * loss_smooth
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        torch.save(
            model.state_dict(),
            f"{checkpoint_dir}/retinex_epoch_{epoch+1}.pth"
        )
