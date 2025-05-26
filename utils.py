import torch
import numpy as np
from torchvision.utils import save_image as tv_save_image
from PIL import Image

# Attempt to use skimage for metrics if available
try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    # This warning will be printed once when utils.py is imported if skimage is not found
    print("Warning: scikit-image not found. PSNR/SSIM calculations will use basic PyTorch implementations. "
          "Install scikit-image for more accurate metrics (pip install scikit-image).")

# --- Image/Tensor Conversion ---

def tensor_to_numpy_image(tensor_img):
    """
    Converts a PyTorch image tensor (C, H, W) in range [0,1] 
    to a NumPy array (H, W, C) in range [0,255] with dtype uint8.
    """
    if not torch.is_tensor(tensor_img):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor_img.ndim != 3:
        # Handle batch of images if necessary, or raise error
        raise ValueError("Input tensor must be 3-dimensional (C, H, W).")
    
    img_np = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np

def save_tensor_image(tensor_img, filepath):
    """
    Saves a PyTorch image tensor (typically C, H, W, range [0,1]) to a file.
    Uses torchvision.utils.save_image.
    Args:
        tensor_img: PyTorch tensor representing the image.
        filepath (str): Path to save the image.
    """
    if not torch.is_tensor(tensor_img):
        raise TypeError("Input must be a PyTorch tensor.")
    tv_save_image(tensor_img, filepath)


# --- Metrics ---

def calculate_psnr(target, output, data_range=1.0):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    Supports both PyTorch tensors and NumPy arrays.
    If scikit-image is available, it's used for NumPy arrays.
    For PyTorch tensors or if scikit-image is unavailable, a basic PyTorch implementation is used.

    Args:
        target: Ground truth image (PyTorch tensor or NumPy array).
        output: Predicted image (PyTorch tensor or NumPy array).
        data_range (float or int): The data range of the input images (e.g., 1.0 for [0,1], 255 for [0,255]).
                                   Note: skimage_psnr uses this directly.
                                   The PyTorch fallback assumes [0, data_range] and normalizes MSE by data_range.
    Returns:
        float: PSNR value in dB.
    """
    if isinstance(target, np.ndarray) and isinstance(output, np.ndarray):
        if SKIMAGE_AVAILABLE:
            # Ensure skimage data_range matches input numpy array range if they are uint8
            if target.dtype == np.uint8:
                skimage_data_range = 255
            else: # Assuming float [0,1] or similar if not uint8
                skimage_data_range = data_range 
            return skimage_psnr(target, output, data_range=skimage_data_range)
        else: # Fallback for NumPy arrays if skimage not available
            # Convert to tensor for consistent fallback calculation
            target = torch.from_numpy(target.astype(np.float32)).permute(2,0,1) / (skimage_data_range if skimage_data_range == 255 else 1.0)
            output = torch.from_numpy(output.astype(np.float32)).permute(2,0,1) / (skimage_data_range if skimage_data_range == 255 else 1.0)
            # now target/output are tensors in [0,1] range, so data_range for pytorch_psnr should be 1.0
            return _pytorch_psnr(target, output, data_range=1.0)
            
    elif torch.is_tensor(target) and torch.is_tensor(output):
        return _pytorch_psnr(target, output, data_range)
    else:
        raise TypeError(f"Inputs must be both PyTorch tensors or both NumPy arrays. "
                        f"Got target: {type(target)}, output: {type(output)}")

def _pytorch_psnr(target_tensor, output_tensor, data_range=1.0):
    """Basic PSNR calculation using PyTorch, assumes tensors are in [0, data_range]."""
    if not (target_tensor.shape == output_tensor.shape):
        raise ValueError("Input tensors must have the same shape.")
    
    mse = torch.mean((target_tensor - output_tensor) ** 2)
    if mse == 0:
        return float('inf')
    
    # Formula: 20 * log10(MAX_I / sqrt(MSE))
    # MAX_I is data_range
    return 20 * torch.log10(data_range / torch.sqrt(mse))


def calculate_ssim(target, output, data_range=1.0, win_size=7, multichannel=True):
    """
    Calculates Structural Similarity Index (SSIM).
    Supports both PyTorch tensors and NumPy arrays.
    If scikit-image is available, it's used for NumPy arrays.
    For PyTorch tensors or if scikit-image is unavailable, a basic PyTorch-based placeholder is used (returns 0.0 with a warning).

    Args:
        target: Ground truth image (PyTorch tensor or NumPy array).
        output: Predicted image (PyTorch tensor or NumPy array).
        data_range (float or int): The data range of the input images.
                                   For skimage, this is important. For PyTorch, the fallback is basic.
        win_size (int): Window size for SSIM calculation (used by skimage).
        multichannel (bool): If True, treat the last dimension of NumPy array as channels (used by skimage).

    Returns:
        float: SSIM value.
    """
    if isinstance(target, np.ndarray) and isinstance(output, np.ndarray):
        if SKIMAGE_AVAILABLE:
            # Ensure skimage data_range matches input numpy array range
            if target.dtype == np.uint8:
                skimage_data_range = 255
                # skimage SSIM expects float inputs for data_range other than 1.
                # Or, if input is uint8, it assumes data_range is 255.
                # If float input, it assumes data_range is 1.0 unless specified.
                # We convert to float here to be safe if data_range is not 255.
                # However, our tensor_to_numpy_image converts to uint8 [0,255]
            else: # Assuming float [0,1] or similar
                skimage_data_range = data_range

            # Scikit-image SSIM can be sensitive to data type for data_range.
            # If input is uint8, it assumes data_range=255. If float, data_range=1.0.
            # It's safer to pass float arrays to skimage.metrics.structural_similarity
            # if we are specifying data_range, unless it's uint8 and data_range=255.
            # Our tensor_to_numpy_image produces uint8.
            return skimage_ssim(target, output, data_range=skimage_data_range, win_size=win_size, 
                                channel_axis=-1 if multichannel else None, # Use channel_axis for multichannel
                                gaussian_weights=True, use_sample_covariance=False) # common defaults
        else:
            # Fallback for NumPy arrays if skimage not available
            print("Warning: calculate_ssim (NumPy) falling back to dummy value as scikit-image is not available.")
            return 0.0 # Placeholder
            
    elif torch.is_tensor(target) and torch.is_tensor(output):
        return _pytorch_ssim_placeholder(target, output, data_range)
    else:
        raise TypeError(f"Inputs must be both PyTorch tensors or both NumPy arrays. "
                        f"Got target: {type(target)}, output: {type(output)}")

def _pytorch_ssim_placeholder(target_tensor, output_tensor, data_range=1.0):
    """Placeholder for SSIM calculation for PyTorch tensors."""
    # A real PyTorch SSIM would be more complex, e.g. from kornia or a direct implementation
    print("Warning: calculate_ssim (Tensor) is a placeholder and returns 0.0. "
          "Consider implementing a PyTorch-based SSIM or ensuring inputs are NumPy for scikit-image.")
    return torch.tensor(0.0) # Dummy value

# --- Dataset Utilities (Placeholder for now, can be expanded) ---
# Example: Common image file checking function
def is_image_file(filename):
    """Checks if a file is a common image type."""
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])

# Could add a BaseImageDataset here in the future if PairedImageDataset and TestImageDataset share more logic.
# class BaseImageDataset(Dataset):
#     def __init__(self, transform=None):
#         self.transform = transform
#         # ... common logic ...

#     def _is_image_file(self, filename):
#         return is_image_file(filename)
#     ...

if __name__ == '__main__':
    # Example Usage (and basic test)
    print(f"scikit-image available: {SKIMAGE_AVAILABLE}")

    # Test tensor_to_numpy_image
    test_tensor = torch.rand(3, 64, 64)
    numpy_img = tensor_to_numpy_image(test_tensor)
    print(f"tensor_to_numpy_image: Input tensor shape {test_tensor.shape}, Output numpy shape {numpy_img.shape}, dtype {numpy_img.dtype}")
    assert numpy_img.shape == (64, 64, 3)
    assert numpy_img.dtype == np.uint8

    # Test PSNR
    t1_tensor = torch.rand(3, 32, 32)
    t2_tensor = t1_tensor + (torch.rand(3, 32, 32) - 0.5) * 0.1
    t2_tensor = torch.clamp(t2_tensor, 0, 1)
    
    psnr_tensor = calculate_psnr(t1_tensor, t2_tensor, data_range=1.0)
    print(f"PSNR (Tensor, [0,1]): {psnr_tensor.item() if torch.is_tensor(psnr_tensor) else psnr_tensor:.2f} dB")

    t1_numpy = tensor_to_numpy_image(t1_tensor) # HWC, uint8, [0,255]
    t2_numpy = tensor_to_numpy_image(t2_tensor) # HWC, uint8, [0,255]
    
    psnr_numpy = calculate_psnr(t1_numpy, t2_numpy, data_range=255) # data_range is important for numpy version
    print(f"PSNR (NumPy, [0,255]): {psnr_numpy:.2f} dB")

    # Test SSIM
    ssim_tensor = calculate_ssim(t1_tensor, t2_tensor, data_range=1.0)
    print(f"SSIM (Tensor, [0,1]): {ssim_tensor.item() if torch.is_tensor(ssim_tensor) else ssim_tensor:.4f}")
    
    ssim_numpy = calculate_ssim(t1_numpy, t2_numpy, data_range=255)
    print(f"SSIM (NumPy, [0,255]): {ssim_numpy:.4f}")

    # Test save_tensor_image
    if not os.path.exists("temp_test_outputs"):
        os.makedirs("temp_test_outputs")
    save_tensor_image(test_tensor, "temp_test_outputs/test_utils_save.png")
    print("Saved test image to temp_test_outputs/test_utils_save.png")
    # Cleanup
    # import shutil
    # shutil.rmtree("temp_test_outputs")
    
    print("Basic utils.py tests completed.")
