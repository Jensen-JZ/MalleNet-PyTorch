# MalleNet-PyTorch

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://jensen.dlab.ac.cn/we-stand-with-ukraine/)

Unofficial PyTorch implementation of *Fast and High-Quality Image Denoising via Malleable Convolutions*. We implemented it based on the author's [official codes](https://yifanjiang.net/MalleConv.html) here, and the official codes are not completed. They are missing many vital parameters and train or test scripts.

### Requirements

* Python >= 3.7
* PyTorch >= 1.8.2 (LTS)
* NumPy >= 1.19.2
* GCC >= 5.0
* ...
* For a full list of Python dependencies and to install them, please use the provided `requirements.txt` file:
  ```shell
  pip install -r requirements.txt
  ```

> **Warning: Experimental C++ Bilateral Slice Layer (Currently Incomplete)**
>
> The C++/CUDA bilateral slice extension described below is **experimental and currently incomplete.** Critical source files (specifically `bilateral_slice.h`, `bilateral_slice_cuda_kernel.cu`, and `bilateral_slice_cpu.cpp`) are **missing** from the `configs` directory.
>
> **Attempting to build this extension using the instructions below will fail.**
>
> The `MalleNet` model provided in `model.py` includes a **PyTorch-native implementation** of the Bilateral Slice operation as a fallback. This ensures that the model is fully runnable using the provided Python scripts (`train.py`, `test.py`) without requiring the C++ extension. Please be aware that this Python-based fallback will have different performance characteristics compared to a potentially optimized C++/CUDA version.

## Experimental C++ Bilateral Slice Layer (Currently Non-Functional)

* Moving the configuration files to the specified path.

  ```shell
  mv configs/* deep_bilateral_network/bilateral_slice_op/
  ```

* Building the bilateral slice layer.

  ```shell
  cd deep_bilateral_network/bilateral_slice_op/
  python setup.py install
  ```

The complete MalleNet code is in the file `model.py`...

## Training and Testing

This project includes scripts to train new MalleNet models and test existing ones.

### Data Preparation

The training and testing scripts expect image files (e.g., `.png`, `.jpg`, `.jpeg`) organized in the following directory structure:

*   **Training Data:**
    *   Noisy images: `path/to/your/train_data/noisy/`
    *   Corresponding ground truth images: `path/to/your/train_data/gt/`
*   **Validation Data (Optional but Recommended):**
    *   Noisy images: `path/to/your/val_data/noisy/`
    *   Corresponding ground truth images: `path/to/your/val_data/gt/`
*   **Testing Data:**
    *   Noisy images: `path/to/your/test_data/noisy/`
    *   Corresponding ground truth images (optional, for metrics): `path/to/your/test_data/gt/`

Images in the `noisy` and `gt` subdirectories are expected to be paired by their sorted filenames (e.g., `001.png` in `noisy` corresponds to `001.png` in `gt`).

### Training (`train.py`)

The `train.py` script is used to train a MalleNet model.

**Example Usage:**

```bash
python train.py \
    --train_noisy_dir path/to/your/train_data/noisy \
    --train_gt_dir path/to/your/train_data/gt \
    --val_noisy_dir path/to/your/val_data/noisy \
    --val_gt_dir path/to/your/val_data/gt \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_workers 4 \
    --log_interval 10 \
    --checkpoint_interval 5
    \
    # MalleNet Architecture Arguments (adjust as needed)
    --in_channel 3 \
    --out_channel 3 \
    --num_feature 64 \
    --low_res 'down' \
    --down_scale 2 \
    --stage 3 \
    --depth 3 
```

**Key Arguments for `train.py`:**

*   `--train_noisy_dir`, `--train_gt_dir`: Paths to training noisy and ground truth images.
*   `--val_noisy_dir`, `--val_gt_dir`: Paths to validation noisy and ground truth images.
*   `--output_dir`: Directory to save model checkpoints.
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--lr`: Learning rate.
*   `--num_workers`: Number of DataLoader workers.
*   `--log_interval`: Print training log every N batches.
*   `--checkpoint_interval`: Save a checkpoint every N epochs.
*   **MalleNet Architecture Arguments:**
    *   `--in_channel`, `--out_channel`: Input/output image channels.
    *   `--num_feature`: Base number of features in convolutional layers.
    *   `--low_res`: Strategy for low-resolution processing in ModelTwo (e.g., 'down', 'downavg2').
    *   `--down_scale`: Downscaling factor in ModelOne's pyramid decomposition.
    *   `--stage`: Number of stages in ModelTwo.
    *   `--depth`: Depth of the pyramid decomposition in ModelOne.
    *   *(It is crucial that these architecture arguments match if you later use a checkpoint with `test.py`)*

### Testing (`test.py`)

The `test.py` script is used to evaluate a trained MalleNet model on a test dataset.

**Example Usage:**

```bash
python test.py \
    --test_noisy_dir path/to/your/test_data/noisy \
    --test_gt_dir path/to/your/test_data/gt \  # Optional: for calculating PSNR/SSIM metrics
    --model_checkpoint ./checkpoints/best_model.pth \
    --output_dir ./results \
    --num_workers 4 \
    \
    # MalleNet Architecture Arguments (MUST match the trained model)
    --in_channel 3 \
    --out_channel 3 \
    --num_feature 64 \
    --low_res 'down' \
    --down_scale 2 \
    --stage 3 \
    --depth 3
```

**Key Arguments for `test.py`:**

*   `--test_noisy_dir`: Path to test noisy images.
*   `--test_gt_dir`: Path to test ground truth images (optional, for metrics).
*   `--model_checkpoint`: Path to the trained model checkpoint (`.pth` file).
*   `--output_dir`: Directory to save denoised images.
*   **MalleNet Architecture Arguments:** These **must** match the arguments used to train the loaded model checkpoint for correct model instantiation.

