import random
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import wandb
from src.data import get_clean_ffhq_dataloaders, get_clean_ffhq_dataloaders_unnormalized
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from eval import compute_metrics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def patchify(x, patch_size):
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    patches = x.reshape(B, h * w, patch_size * patch_size * C)
    return patches

def unpatchify(x, patch_size, img_size):
    B, N, patch_dim = x.shape
    H, W = img_size, img_size
    h = H // patch_size
    w = W // patch_size
    x = x.reshape(B, h, w, patch_size, patch_size, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)
    imgs = x.reshape(B, 3, H, W)
    return imgs

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

class MAE(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        encoder_dim=512,  # Reduced from 768
        decoder_dim=256,  # Reduced from 768
        encoder_depth=6,  # Reduced from 16
        decoder_depth=4,  # Reduced from 12
        encoder_heads=16,      # Reduced from 16
        decoder_heads=8,      # Reduced from 16
        mask_ratio=0.75,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        
        print(f"MAE Config: img_size={img_size}, patch_size={patch_size}, num_patches={self.num_patches}")
        print(f"Encoder: dim={encoder_dim}, depth={encoder_depth}, heads={encoder_heads}")
        print(f"Decoder: dim={decoder_dim}, depth={decoder_depth}, heads={decoder_heads}")
        print(f"Mask ratio: {mask_ratio}")
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, encoder_dim)

        # Positional embeddings - use smaller initialization
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim) * 0.02)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))


        # Simpler Encoder - using basic transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,  # Start with no dropout
            activation='gelu',
            batch_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, encoder_depth)
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # Projection for decoder input
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        # Simpler Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Sequential(
            nn.Linear(decoder_dim, patch_size**2 * 3),
            nn.Sigmoid()          # clamp to [0, 1]
        )

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # Patchify input
        x = self.patch_embed(imgs)  # (B, N, D)
        x = x + self.encoder_pos_embed

        # Apply masking
        x_masked, mask, ids_restore = self.random_masking(x)

        # Encode visible patches
        x_encoded = self.encoder_blocks(x_masked)
        x_encoded = self.encoder_norm(x_encoded)

        # Prepare decoder input
        x_dec = self.enc_to_dec(x_encoded)
        B, L_visible, D = x_dec.shape

        # Add mask tokens
        mask_tokens = self.mask_token.repeat(B, self.num_patches - L_visible, 1)
        x_combined = torch.cat([x_dec, mask_tokens], dim=1)

        # Unshuffle
        x_unshuffled = torch.gather(x_combined, dim=1, 
                                   index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # Add positional embeddings
        x_unshuffled = x_unshuffled + self.decoder_pos_embed

        # Decode
        x_decoded = self.decoder_blocks(x_unshuffled)
        x_decoded = self.decoder_norm(x_decoded)
        pred = self.head(x_decoded)

        return pred, mask

        # Recommended loss function
    def loss(self, pred, target, mask):
        """
        Computes the Mean Squared Error on the masked patches.
        pred: [B, N, patch_size**2 * 3]
        target: [B, 3, H, W]
        mask: [B, N], 0 for visible, 1 for masked
        """
        target_patches = patchify(target, self.patch_size)
        
        # Calculate MSE loss
        loss = (pred - target_patches) ** 2
        loss = loss.mean(dim=-1)  # Loss per patch
        
        # Only keep the loss on masked patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        
        return loss

    def reconstruct(self, pred):
        """Convert predictions back to image format"""
        return unpatchify(pred, self.patch_size, self.img_size)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}

    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            
            # Get model predictions
            pred_patches, mask = model(imgs)
            loss = model.loss(pred_patches, imgs, mask)
            total_loss += loss.item()
            
            # Reconstruct images from patches
            reconstructed_imgs = model.reconstruct(pred_patches)
            
            # Clamp values to [0, 1] range for metric calculation
            reconstructed_imgs = torch.clamp(reconstructed_imgs, 0, 1)

            # --- Compute metrics for each image in the batch ---
            for i in range(imgs.shape[0]):
                gt_img = imgs[i]
                pred_img = reconstructed_imgs[i]
                
                # Use your existing function to get metrics for one image
                metrics = compute_metrics(gt_img, pred_img)
                
                # Store the results
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])

    # Calculate the average of all collected metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return total_loss / len(dataloader), avg_metrics


def train_one_epoch(model, dataloader, optimizer, device, epoch, global_step, verbose=True):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        
        # Forward pass
        pred, mask = model(imgs)
        loss = model.loss(pred, imgs, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        wandb.log({"train_batch_loss": batch_loss})
        global_step +=1

        
        # Print detailed info for first few batches
        if batch_idx < 3 or batch_idx % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}")
            print(f"    Loss: {batch_loss:.4f}")
            print(f"    Grad norm: {total_norm:.4f}")
            print(f"    Mask ratio actual: {mask.mean().item():.3f}")
            print(f"    Pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
            print(f"    Target range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
            
        # Save reconstruction sample for first batch of first epoch
        if batch_idx == 0:
            save_reconstruction_sample(model, imgs, pred, mask, f"/home/hice1/avarma49/scratch/reconstruction/reconstruction_epoch_{epoch}.png", epoch + 1)
    
    return total_loss / num_batches

def save_reconstruction_sample(model, imgs, pred, mask, filename, epoch):
    model.eval()
    with torch.no_grad():
        img = imgs[0:1]
        reconstructed = model.reconstruct(pred[0:1])
        
        # Create masked version
        masked_img = img.clone()
        
        # Dynamically calculate patch grid size
        num_patches = mask.shape[1]  # Should be 256 for 256x256 images
        patches_per_side = int(num_patches ** 0.5)  # sqrt(256) = 16
        patch_size = model.patch_size  # Should be 16
        
        # Reshape mask to 2D grid
        mask_2d = mask[0].reshape(patches_per_side, patches_per_side)
        
        # Apply mask to image
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                if mask_2d[i, j] == 1:  # Masked patch
                    start_h = i * patch_size
                    start_w = j * patch_size
                    end_h = start_h + patch_size
                    end_w = start_w + patch_size
                    
                    # Set masked patches to gray (0.5)
                    masked_img[0, :, start_h:end_h, start_w:end_w] = 0.5
        
        # Convert to numpy and transpose for visualization
        original = img[0].cpu().permute(1, 2, 0).numpy()
        masked = masked_img[0].cpu().permute(1, 2, 0).numpy()
        recon = reconstructed[0].cpu().permute(1, 2, 0).numpy()
        
        # Clip to valid range [0, 1]
        original = np.clip(original, 0, 1)
        masked = np.clip(masked, 0, 1)
        recon = np.clip(recon, 0, 1)
        
        # Create 3-panel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(masked)
        axes[1].set_title(f'Masked Input ({model.mask_ratio:.1%} masked)', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(recon)
        axes[2].set_title('Reconstructed', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        wandb.log({"Reconstruction Example": wandb.Image(filename, caption=f"Epoch {epoch}")})
        
        print(f"Saved reconstruction sample to {filename}")
        print(f"Patch grid: {patches_per_side}x{patches_per_side}, Patch size: {patch_size}x{patch_size}")
        print(f"Masked patches: {mask[0].sum().item()}/{num_patches} ({mask[0].sum().item()/num_patches:.1%})")
# Training setup with debugging
# Training setup with debugging
def main(resume_from_checkpoint=None):
    global_step = 0
    config = {
        "epochs": 100,
        "batch_size": 512, # Or whatever you decide on
        "lr": 1e-4,
        "weight_decay": 0.05,
        "img_size": 256,
        "patch_size": 16,
        "encoder_dim": 1024,
        "decoder_dim": 512,
        "encoder_depth": 16,
        "decoder_depth": 8,
        "encoder_heads": 16,
        "decoder_heads": 8,
        "mask_ratio": 0.75
    }
    wandb.init(
        project="mae-ffhq-reconstruction",  # A name for your project
        config=config                      # The hyperparameter configuration
    )


    total_epochs = wandb.config.epochs

    model = MAE(
        img_size=wandb.config.img_size,
        patch_size=wandb.config.patch_size,
        encoder_dim=wandb.config.encoder_dim,  # Much smaller
        decoder_dim=wandb.config.decoder_dim,
        encoder_depth=wandb.config.encoder_depth,
        decoder_depth=wandb.config.decoder_depth,
        encoder_heads=wandb.config.encoder_heads,
        decoder_heads=wandb.config.decoder_heads,
        mask_ratio=wandb.config.mask_ratio,
    )

    # Load your data
    train_loader, test_loader = get_clean_ffhq_dataloaders_unnormalized(
        "/home/hice1/avarma49/scratch/ffhq-256/",
        batch_size=wandb.config.batch_size  # Smaller batch size
    )

    optimizer = AdamW(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-10)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])



    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    wandb.watch(model, log="gradients", log_freq=100) # log_freq logs every 100 batches


    # Resume from checkpoint if provided
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'psnr': [],
        'ssim': [],
        'lpips': []
    }


    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        # Load previous losses if they exist
        try:
            with open("debug_losses.json", "r") as f:
                losses = json.load(f)
            print(f"Loaded {len(losses)} previous loss values")
        except FileNotFoundError:
            print("No previous loss file found, starting fresh loss tracking")

        print(f"Resuming from epoch {start_epoch + 1}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Training on {device}")

    epochs_to_run = total_epochs - start_epoch

    # warmup_epochs = 10 # A 10-epoch warmup is a safe choice
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, global_step, verbose=True)

        val_loss, val_metrics = validate(model, test_loader, device)

        scheduler.step()

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Metrics - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, LPIPS: {val_metrics['lpips']:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "psnr": val_metrics['psnr'],
            "ssim": val_metrics['ssim'],
            "lpips": val_metrics['lpips'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['psnr'].append(float(val_metrics['psnr']))
        history['ssim'].append(float(val_metrics['ssim']))
        history['lpips'].append(float(val_metrics['lpips']))


        # Save progress
        with open("metrics_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"/home/hice1/avarma49/scratch/checkpoints/debug_checkpoint_epoch_{epoch + 1}.pt")

    wandb.finish()

    return model, losses


if __name__ == "__main__":
    model, losses = main()
    