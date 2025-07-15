import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np
from eval import *
import torchvision.utils as vutils
from PIL import Image
import os

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to("cuda").eval()
def compute_metrics(gt_img, pred_img):
    # Convert to numpy (H, W, C) for SSIM & PSNR
    gt = gt_img.permute(1, 2, 0).cpu().numpy()
    pred = pred_img.permute(1, 2, 0).cpu().numpy()

    # Compute PSNR and SSIM (for color image)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_score, _ = ssim(gt, pred, channel_axis=-1, data_range=1.0, full=True)

    # Convert to [-1, 1] for LPIPS
    gt_lpips = 2 * gt_img - 1
    pred_lpips = 2 * pred_img - 1
    lpips_score = lpips_model(gt_lpips.cuda().unsqueeze(0), pred_lpips.cuda().unsqueeze(0)).item()

    return {'psnr': psnr, 'ssim': ssim_score, 'lpips': lpips_score}

def save_visual_comparison(gt, noisy, recon, path):
    grid = vutils.make_grid(torch.cat([gt, noisy, recon], dim=0), nrow=3)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(ndarr).save(path)

def evaluate_model(model, dataloader, masking_fn, noise_schedule_fn, arcface_model=None, save_dir=None):
    results = []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for i, batch in enumerate(dataloader):
        x = batch['image'].cuda()
        gt = batch['gt'].cuda()

        mask = masking_fn(x)
        noisy_x = noise_schedule_fn(x, mask)

        with torch.no_grad():
            recon = model(noisy_x, mask)

        metrics = compute_metrics(gt[0], recon[0])

        if arcface_model is not None:
            emb_gt = arcface_model(gt)
            emb_recon = arcface_model(recon)
            identity_sim = torch.nn.functional.cosine_similarity(emb_gt, emb_recon).item()
            metrics['identity'] = identity_sim

        if save_dir:
            save_visual_comparison(gt[0], noisy_x[0], recon[0], os.path.join(save_dir, f'example_{i}.png'))

        results.append(metrics)

    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return avg

def run_ablation(model, dataloader, masking_strategies, noise_schedules, arcface_model=None):
    for mask_name, mask_fn in masking_strategies.items():
        for noise_name, noise_fn in noise_schedules.items():
            print(f"Running {mask_name} mask with {noise_name} noise...")
            save_dir = f"results/{mask_name}_{noise_name}"
            metrics = evaluate_model(model, dataloader, mask_fn, noise_fn, arcface_model, save_dir=save_dir)
            print(f"{mask_name} + {noise_name}:", metrics)
