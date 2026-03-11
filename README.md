# GAN vs StyleGAN 3 POC

Proof of Concept comparing basic GAN with StyleGAN 3.

## Files

- gan_mnist.py - Basic GAN implementation on MNIST
- generate_gan_images.py - Generate images using trained GAN
- compare_gans.py - Compare GAN and StyleGAN 3 outputs
- models/gan_mnist_generator.pth - Trained GAN generator weights
- examples/ - Sample output images

## Setup

1. Install PyTorch with CUDA support:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

2. Install required packages:
   pip install numpy matplotlib pillow tqdm

3. Clone StyleGAN 3 repository:
   git clone https://github.com/NVlabs/stylegan3.git

4. Install StyleGAN 3 dependencies:
   pip install click scipy ninja imageio imageio-ffmpeg pyopengl glfw pyspng

5. Modify StyleGAN 3 code to use reference implementation:
   Change default impl parameter from 'cuda' to 'ref' in:
   - stylegan3/torch_utils/ops/bias_act.py
   - stylegan3/torch_utils/ops/filtered_lrelu.py
   - stylegan3/torch_utils/ops/upfirdn2d.py

## Usage

1. Train Basic GAN:
   python gan_mnist.py

2. Generate GAN Images:
   python generate_gan_images.py

3. Generate StyleGAN 3 Images:
   python stylegan3/gen_images.py --outdir=stylegan3_output --trunc=1 --seeds=0-15 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

4. Compare Results:
   python compare_gans.py

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
