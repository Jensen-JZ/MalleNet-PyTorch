import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# Assuming MalleNet is in model.py
from model import MalleNet 
# For PSNR/SSIM, can use libraries like skimage later if needed
from utils import calculate_psnr, calculate_ssim, is_image_file # Import is_image_file if used by dataset

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train MalleNet Model")
    parser.add_argument('--train_noisy_dir', type=str, required=True, help='Path to noisy training images')
    parser.add_argument('--train_gt_dir', type=str, required=True, help='Path to ground truth training images')
    parser.add_argument('--val_noisy_dir', type=str, help='Path to noisy validation images')
    parser.add_argument('--val_gt_dir', type=str, help='Path to ground truth validation images')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=10, help='Print log every N batches')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    # MalleNet specific parameters (based on model.py defaults, can be exposed as args)
    parser.add_argument('--in_channel', type=int, default=3, help='Input channels for MalleNet')
    parser.add_argument('--out_channel', type=int, default=3, help='Output channels for MalleNet')
    parser.add_argument('--num_feature', type=int, default=64, help='Number of features in MalleNet')
    parser.add_argument('--low_res', type=str, default='down', help='Low resolution processing mode for MalleNet')
    parser.add_argument('--down_scale', type=int, default=2, help='Down scale factor for MalleNet ModelOne')
    parser.add_argument('--stage', type=int, default=3, help='Number of stages in MalleNet ModelTwo')
    parser.add_argument('--depth', type=int, default=3, help='Depth for MalleNet ModelOne')

    return parser.parse_args()

# --- Dataset Class ---
class PairedImageDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.transform = transform

        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if self._is_image_file(f)])
        self.gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if self._is_image_file(f)])

        if len(self.noisy_files) != len(self.gt_files):
            raise ValueError("Number of noisy and ground truth images must be the same.")
        if len(self.noisy_files) == 0:
            raise ValueError(f"No images found in {noisy_dir} or {gt_dir}")

    def _is_image_file(self, filename):
        return is_image_file(filename) # Use from utils

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_img_path = self.noisy_files[idx]
        gt_img_path = self.gt_files[idx] # Assuming direct pairing by sorted order

        # To ensure pairing is correct, could implement more robust name matching logic
        # e.g. noisy_001.png and gt_001.png
        # For now, relies on sorted lists having corresponding pairs.

        noisy_img = Image.open(noisy_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')

        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img

# --- Main Training Function ---
def main(args):
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL image [0,255] to Tensor [0,1]
        # Normalize can be added if needed, e.g. transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # But MalleNet output is clipped to [0,1], so direct [0,1] input is fine.
    ])

    # DataLoaders
    print("Loading training data...")
    train_dataset = PairedImageDataset(args.train_noisy_dir, args.train_gt_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print(f"Loaded {len(train_dataset)} training images.")

    val_loader = None
    if args.val_noisy_dir and args.val_gt_dir:
        print("Loading validation data...")
        val_dataset = PairedImageDataset(args.val_noisy_dir, args.val_gt_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"Loaded {len(val_dataset)} validation images.")
    else:
        print("No validation data provided.")

    # Model
    print("Initializing MalleNet model...")
    # These MalleNet parameters can be exposed via argparse if desired
    model = MalleNet(
        in_channel=args.in_channel,
        num_feature=args.num_feature,
        out_channel=args.out_channel,
        low_res=args.low_res,
        down_scale=args.down_scale,
        stage=args.stage,
        depth=args.depth
    ).to(device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss().to(device) # MAE Loss

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_psnr = 0.0

    # Training Loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (noisy_imgs, gt_imgs) in enumerate(train_loader):
            noisy_imgs, gt_imgs = noisy_imgs.to(device), gt_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, gt_imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}] completed. Average Training Loss: {avg_epoch_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0 # Placeholder
            with torch.no_grad():
                for noisy_val, gt_val in val_loader:
                    noisy_val, gt_val = noisy_val.to(device), gt_val.to(device)
                    outputs_val = model(noisy_val)
                    val_loss += criterion(outputs_val, gt_val).item()
                    
                    # Calculate PSNR for each item in batch and average
                    # Ensure inputs to calculate_psnr are appropriate (e.g., tensors in [0,1] range)
                    for i in range(outputs_val.size(0)):
                        # PSNR from utils.py can handle tensors directly
                        current_psnr = calculate_psnr(gt_val[i], outputs_val[i], data_range=1.0)
                        if torch.is_tensor(current_psnr):
                             val_psnr += current_psnr.item()
                        else: # if it returns float (e.g. inf)
                             val_psnr += current_psnr
                        
                        # SSIM from utils.py (placeholder for tensors)
                        current_ssim = calculate_ssim(gt_val[i], outputs_val[i], data_range=1.0)
                        if torch.is_tensor(current_ssim):
                            val_ssim += current_ssim.item()
                        else:
                            val_ssim += current_ssim

            avg_val_loss = val_loss / len(val_loader)
            avg_val_psnr = val_psnr / len(val_dataset) # Average PSNR over all images
            avg_val_ssim = val_ssim / len(val_dataset) # Average SSIM
            print(f"Epoch [{epoch}/{args.epochs}] Validation: Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}")

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')

                # Also save based on SSIM if that's a key metric, or a combined score
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to {best_model_path} (PSNR: {best_val_psnr:.2f} dB)")

        # Save checkpoints
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        latest_model_path = os.path.join(args.output_dir, 'latest_model.pth')
        torch.save(model.state_dict(), latest_model_path)

    print("Training finished.")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB (if validation was performed)")
    print(f"Latest model saved to {os.path.join(args.output_dir, 'latest_model.pth')}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
