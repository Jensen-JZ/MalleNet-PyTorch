import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import MalleNet # Assuming MalleNet is in model.py
from utils import (
    calculate_psnr, 
    calculate_ssim, 
    tensor_to_numpy_image, 
    save_tensor_image, 
    is_image_file,
    SKIMAGE_AVAILABLE # Import to check if it's available for messages
)

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Test MalleNet Model")
    parser.add_argument('--test_noisy_dir', type=str, required=True, help='Path to noisy test images')
    parser.add_argument('--test_gt_dir', type=str, help='Path to ground truth test images (for metrics)')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save denoised images')
    
    # MalleNet specific parameters (must match the trained model)
    parser.add_argument('--in_channel', type=int, default=3, help='Input channels for MalleNet')
    parser.add_argument('--out_channel', type=int, default=3, help='Output channels for MalleNet (usually same as in_channel)')
    parser.add_argument('--num_feature', type=int, default=64, help='Number of features in MalleNet')
    parser.add_argument('--low_res', type=str, default='down', help='Low resolution processing mode for MalleNet')
    parser.add_argument('--down_scale', type=int, default=2, help='Down scale factor for MalleNet ModelOne')
    parser.add_argument('--stage', type=int, default=3, help='Number of stages in MalleNet ModelTwo')
    parser.add_argument('--depth', type=int, default=3, help='Depth for MalleNet ModelOne')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (typically 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    return parser.parse_args()

# --- Dataset Class ---
class TestImageDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir=None, transform=None):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.transform = transform

        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if is_image_file(f)]) # Use from utils
        
        if not self.noisy_files:
            raise FileNotFoundError(f"No images found in {noisy_dir}")

        if self.gt_dir:
            self.gt_files = sorted([f for f in os.listdir(gt_dir) if is_image_file(f)]) # Use from utils
            if not self.gt_files:
                raise FileNotFoundError(f"No images found in {gt_dir}")
            if len(self.noisy_files) != len(self.gt_files):
                print(f"Warning: Number of noisy files ({len(self.noisy_files)}) and GT files ({len(self.gt_files)}) do not match. Metrics will be calculated for paired files based on sorted order.")
                min_len = min(len(self.noisy_files), len(self.gt_files))
                self.noisy_files = self.noisy_files[:min_len]
                self.gt_files = self.gt_files[:min_len]

    def _is_image_file(self, filename): # This method can be removed if is_image_file from utils is used directly in constructor
        return is_image_file(filename)

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_filename = self.noisy_files[idx]
        noisy_img_path = os.path.join(self.noisy_dir, noisy_filename)
        noisy_img = Image.open(noisy_img_path).convert('RGB')

        if self.transform:
            noisy_img_tensor = self.transform(noisy_img)

        if self.gt_dir and idx < len(self.gt_files): # Ensure GT file exists for this index
            # Basic pairing: assumes noisy_files[idx] corresponds to gt_files[idx]
            # More robust pairing would match filenames (e.g. noisy_001.png -> gt_001.png)
            gt_filename = self.gt_files[idx] 
            gt_img_path = os.path.join(self.gt_dir, gt_filename)
            gt_img = Image.open(gt_img_path).convert('RGB')
            if self.transform:
                gt_img_tensor = self.transform(gt_img)
            return noisy_img_tensor, gt_img_tensor, noisy_filename
        else:
            return noisy_img_tensor, noisy_filename

# --- Main Testing Function ---
def main(args):
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL image [0,255] to Tensor [0,1]
    ])

    # DataLoaders
    print("Loading test data...")
    try:
        test_dataset = TestImageDataset(args.test_noisy_dir, args.test_gt_dir, transform=transform)
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Loaded {len(test_dataset)} test images.")

    # Model
    print("Initializing MalleNet model...")
    model = MalleNet(
        in_channel=args.in_channel,
        num_feature=args.num_feature,
        out_channel=args.out_channel,
        low_res=args.low_res,
        down_scale=args.down_scale,
        stage=args.stage,
        depth=args.depth
    )
    
    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Model checkpoint not found at {args.model_checkpoint}")
        return
    
    print(f"Loading model weights from {args.model_checkpoint}")
    # Load weights, being careful about DataParallel or DistributedDataParallel prefixes
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # Handle potential keys like 'state_dict' or direct weights
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint: # Common in some saving patterns
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if model was saved from DataParallel or DDP
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to {args.output_dir}")

    total_psnr = 0.0
    total_ssim = 0.0
    num_images_for_metrics = 0

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(test_loader):
            if args.test_gt_dir:
                noisy_imgs, gt_imgs, noisy_filenames = data_batch
                gt_imgs = gt_imgs.to(device)
            else:
                noisy_imgs, noisy_filenames = data_batch
            
            noisy_imgs = noisy_imgs.to(device)

            outputs = model(noisy_imgs)
            outputs = torch.clamp(outputs, 0.0, 1.0) # Ensure output is in [0,1]

            for i in range(outputs.size(0)):
                output_img_tensor = outputs[i]
                noisy_filename = noisy_filenames[i]
                
                save_path = os.path.join(args.output_dir, noisy_filename)
                save_tensor_image(output_img_tensor, save_path) # Use from utils

                if (batch_idx * args.batch_size + i + 1) % 10 == 0:
                    print(f"Processed and saved: {save_path}")

                if args.test_gt_dir:
                    gt_img_tensor = gt_imgs[i]
                    
                    # For metrics, utils.calculate_psnr/ssim can take tensors or numpy arrays.
                    # If SKIMAGE_AVAILABLE is true in utils, and we want to use it, 
                    # we should convert to numpy. Otherwise, tensor versions will be used.
                    if SKIMAGE_AVAILABLE: # Prefer skimage for numpy arrays if available
                        output_np = tensor_to_numpy_image(output_img_tensor) # from utils
                        gt_np = tensor_to_numpy_image(gt_img_tensor) # from utils
                        current_psnr = calculate_psnr(gt_np, output_np, data_range=255)
                        current_ssim = calculate_ssim(gt_np, output_np, data_range=255)
                    else: # Fallback to PyTorch based calculations in utils.py
                          # These expect tensors in [0,1] range
                        current_psnr = calculate_psnr(gt_img_tensor, output_img_tensor, data_range=1.0)
                        current_ssim = calculate_ssim(gt_img_tensor, output_img_tensor, data_range=1.0)
                    
                    # Ensure current_psnr/ssim are float/int before summing
                    total_psnr += current_psnr.item() if torch.is_tensor(current_psnr) else current_psnr
                    total_ssim += current_ssim.item() if torch.is_tensor(current_ssim) else current_ssim
                    num_images_for_metrics += 1
                    print(f"  Metrics for {noisy_filename}: PSNR: {total_psnr/num_images_for_metrics:.2f} dB, SSIM: {total_ssim/num_images_for_metrics:.4f}")


    print("Testing finished.")
    if args.test_gt_dir and num_images_for_metrics > 0:
        avg_psnr = total_psnr / num_images_for_metrics
        avg_ssim = total_ssim / num_images_for_metrics
        print(f"Average PSNR over {num_images_for_metrics} images: {avg_psnr:.2f} dB")
        print(f"Average SSIM over {num_images_for_metrics} images: {avg_ssim:.4f}")
    elif args.test_gt_dir:
        print("Ground truth directory was provided, but no images were processed for metrics (check dataset pairing or file counts).")


if __name__ == '__main__':
    args = parse_args()
    main(args)
