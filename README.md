[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://jensen.dlab.ac.cn/we-stand-with-ukraine/)

# MalleNet-PyTorch

Unofficial PyTorch implementation of *Fast and High-Quality Image Denoising via Malleable Convolutions*. We implemented it based on the author's [official codes](https://yifanjiang.net/MalleConv.html) here, and the official codes are not completed. They are missing many vital parameters and train or test scripts.

### Requirements

* Python >= 3.7
* PyTorch >= 1.8.2 (LTS)
* NumPy >= 1.19.2
* GCC >= 5.0
* ...



## Usage

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

