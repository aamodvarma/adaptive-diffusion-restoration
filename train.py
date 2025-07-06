import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
        num_heads=8,      # Reduced from 16
        mask_ratio=0.75,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        
        print(f"MAE Config: img_size={img_size}, patch_size={patch_size}, num_patches={self.num_patches}")
        print(f"Encoder: dim={encoder_dim}, depth={encoder_depth}, heads={num_heads}")
        print(f"Decoder: dim={decoder_dim}, depth={decoder_depth}")
        print(f"Mask ratio: {mask_ratio}")
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, encoder_dim)

        # Positional embeddings - use smaller initialization
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim) * 0.02)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        # Simpler Encoder - using basic transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
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
            nhead=8,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size**2 * 3)

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

    def loss(self, pred, target, mask):
        """Simple MSE loss on masked patches"""
        target_patches = patchify(target, self.patch_size)
        loss = (pred - target_patches) ** 2
        loss = loss.mean(dim=-1)  # Average over patch dimension
        loss = (loss * mask).sum() / mask.sum()  # Only masked patches
        return loss

    def reconstruct(self, pred):
        """Convert predictions back to image format"""
        return unpatchify(pred, self.patch_size, self.img_size)

def train_one_epoch(model, dataloader, optimizer, device, epoch, verbose=True):
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
            save_reconstruction_sample(model, imgs, pred, mask, f"/home/hice1/avarma49/scratch/reconstruction/reconstruction_epoch_{epoch}.png")
    
    return total_loss / num_batches

def save_reconstruction_sample(model, imgs, pred, mask, filename):
    """Save a sample reconstruction for visualization"""
    model.eval()
    with torch.no_grad():
        # Take first image from batch
        img = imgs[0:1]  # Keep batch dimension
        
        # Reconstruct
        reconstructed = model.reconstruct(pred[0:1])
        
        # Convert to numpy and denormalize
        original = img[0].cpu().permute(1, 2, 0).numpy()
        recon = reconstructed[0].cpu().permute(1, 2, 0).numpy()
        
        # Clip to valid range
        original = np.clip(original, 0, 1)
        recon = np.clip(recon, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(recon)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction sample: {filename}")

# Training setup with debugging
# Training setup with debugging
def main(resume_from_checkpoint=None):
    # Smaller model for debugging
    model = MAE(
        img_size=128,
        patch_size=16,
        encoder_dim=256,  # Much smaller
        decoder_dim=128,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=8,
        mask_ratio=0.75
    )

    # Load your data
    from src.data import get_clean_ffhq_dataloaders
    train_loader, test_loader = get_clean_ffhq_dataloaders(
        "/home/hice1/avarma49/scratch/ffhq-128/",
        batch_size=32  # Smaller batch size
    )

    # Optimizer with very conservative settings
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    losses = []

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

    total_epochs = 100  # Changed from 20 to 50
    epochs_to_run = total_epochs - start_epoch

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, verbose=True)
        scheduler.step()

        print(f"Epoch {epoch + 1} Average Loss: {loss:.4f}")
        losses.append(loss)

        # Save progress
        with open("debug_losses.json", "w") as f:
            json.dump(losses, f, indent=2)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"/home/hice1/avarma49/scratch/checkpoints/debug_checkpoint_epoch_{epoch + 1}.pt")

    return model, losses


if __name__ == "__main__":
    model, losses = main(resume_from_checkpoint="/home/hice1/avarma49/scratch/checkpoints/debug_checkpoint_epoch_50.pt")
