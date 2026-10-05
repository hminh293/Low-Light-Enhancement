import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips


# --------------------------------------------------
# Utility: extract enhanced image from model output
# --------------------------------------------------
def get_pred_tensor(model_output):
    """
    Extract enhanced image tensor from model output.

    Supports:
    - Tensor (GAN, MIRNetV2)
    - Tuple (Retinex-Net, some GAN variants)

    Returns:
        torch.Tensor: enhanced image tensor (B, C, H, W)
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(preds, targets, model_lpips):
    """
    Compute PSNR, SSIM and LPIPS between predictions and targets.
    Assumes input range [0, 1].
    """
    val_psnr = psnr(preds, targets, data_range=1.0)
    val_ssim = ssim(preds, targets, data_range=1.0)

    # LPIPS expects input in [-1, 1]
    preds_lpips = preds * 2.0 - 1.0
    targets_lpips = targets * 2.0 - 1.0

    val_lpips = model_lpips(preds_lpips, targets_lpips).mean()

    return {
        "PSNR": val_psnr.item(),
        "SSIM": val_ssim.item(),
        "LPIPS": val_lpips.item(),
    }


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
def evaluate_model(
    model,
    data_loader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Evaluate enhancement model using PSNR / SSIM / LPIPS.
    Compatible with GAN, Retinex-Net, MIRNetV2.
    """
    model.eval()
    model.to(device)

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0

    model_lpips = lpips.LPIPS(net="vgg").to(device)

    eval_bar = tqdm(data_loader, desc="Evaluating")

    with torch.no_grad():
        for low_light, gt in eval_bar:
            low_light = low_light.to(device)
            gt = gt.to(device)

            output = model(low_light)
            pred = get_pred_tensor(output)

            metrics = compute_metrics(pred, gt, model_lpips)

            psnr_sum += metrics["PSNR"]
            ssim_sum += metrics["SSIM"]
            lpips_sum += metrics["LPIPS"]

    num_samples = len(data_loader)
    print(
        f"PSNR:  {psnr_sum / num_samples:.4f}\n"
        f"SSIM:  {ssim_sum / num_samples:.4f}\n"
        f"LPIPS: {lpips_sum / num_samples:.4f}"
    )


# --------------------------------------------------
# Visualization
# --------------------------------------------------
def draw_images(
    model,
    data_loader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_images=5,
):
    """
    Visualize low-light input, enhanced output and ground truth.
    Compatible with GAN, Retinex-Net, MIRNetV2.
    """
    model.eval()
    model.to(device)

    images_drawn = 0

    with torch.no_grad():
        for low_light, gt in data_loader:
            low_light = low_light.to(device)
            gt = gt.to(device)

            output = model(low_light)
            pred = get_pred_tensor(output)

            for i in range(low_light.size(0)):
                if images_drawn >= num_images:
                    return

                low_img = np.transpose(low_light[i].cpu().numpy(), (1, 2, 0))
                pred_img = np.transpose(pred[i].cpu().numpy(), (1, 2, 0))
                gt_img = np.transpose(gt[i].cpu().numpy(), (1, 2, 0))

                low_img = np.clip(low_img, 0, 1)
                pred_img = np.clip(pred_img, 0, 1)
                gt_img = np.clip(gt_img, 0, 1)

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.title("Low-Light Input")
                plt.imshow(low_img)
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title("Enhanced Output")
                plt.imshow(pred_img)
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title("Ground Truth")
                plt.imshow(gt_img)
                plt.axis("off")

                plt.show()
                images_drawn += 1
