# GAN vs StyleGAN 3 POC

Proof of Concept comparing basic GAN with StyleGAN 3.

## Files

- gan_mnist.py - Basic GAN implementation on MNIST
- generate_gan_images.py - Generate images using trained GAN
- compare_gans.py - Compare GAN and StyleGAN 3 outputs
- models/gan_mnist_generator.pth - Trained GAN generator weights
- examples/ - Sample output images
- stylegan3/ - StyleGAN 3 source code (from NVlabs/stylegan3)

## Setup

1. Install PyTorch with CUDA support:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

2. Install required packages:
   pip install numpy matplotlib pillow tqdm

3. Install StyleGAN 3 dependencies:
   pip install click scipy ninja imageio imageio-ffmpeg pyopengl glfw pyspng

Note: StyleGAN 3 source code is included in this repository. The code has been modified to use reference implementation (no CUDA compilation required).

## Usage

1. Generate GAN Images (using pre-trained model):
   python generate_gan_images.py

2. Generate StyleGAN 3 Images:
   python stylegan3/gen_images.py --outdir=stylegan3_output --trunc=1 --seeds=0-15 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

3. Compare Results:
   python compare_gans.py

4. Optional - Train Basic GAN from scratch:
   python gan_mnist.py

## Results

Basic GAN (MNIST):
- Training Time: ~2 minutes on RTX 4050
- Image Quality: Low-resolution (28x28), blurry digits

StyleGAN 3 (FFHQ):
- Inference Time: ~30 seconds for 16 images
- Image Quality: High-resolution (1024x1024), sharp facial features

## Limitations

- RTX 4050 (6GB VRAM) insufficient for training StyleGAN 3 from scratch
- Different datasets used for comparison
