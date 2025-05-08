import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from glob import glob
from scipy.stats import entropy
import jittor as jt
from jittor import transform
from jittor.dataset import MNIST

from utils.evaluate import evaluate_metrics

@jt.no_grad()
def generate_images(gaussian_diffusion, model, train_loader, batch_size=64, channels=1, image_size=28, n_class=10, w=2, ddim_timesteps=50):
    model.eval()
    gen_dir = './image/generated_images'
    os.makedirs(gen_dir, exist_ok=True)

    # DDPM smpling
    print(f"start DDPM sampling...")
    generated_images = gaussian_diffusion.sample( 
        model, image_size, batch_size=batch_size, channels=channels, n_class=n_class, w=w, mode='random', clip_denoised=True
    )

    if isinstance(generated_images, list):
        final_ddim_images = generated_images[-1]
    else: 
        final_ddim_images = generated_images

    save_images(final_ddim_images, gen_dir, prefix='ddim_eval_samples')

    num_eval_images = final_ddim_images.shape[0] if hasattr(final_ddim_images, 'shape') else batch_size
    fid_score, is_score = evaluate_metrics(gen_dir, real_status_path="./dataset/mnist_train.npz", num_images=num_eval_images)

    return fid_score, is_score



def save_images(images, directory, prefix):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, img in enumerate(images):
        img_np_view = img.data
        img_np = np.array(img_np_view)
        if img_np.ndim == 3 and img_np.shape[0] == 1: 
            img_np = img_np.squeeze(0)

        normalized_img = (img_np + 1) / 2.0
        img_uint8 = (normalized_img * 255).astype(np.uint8)

        try:
            imageio.imwrite(os.path.join(directory, f"{prefix}_{i:04d}.png"), img_uint8)
        except Exception as e:
            print(f"Error saving image {prefix}_{i:04d}.png: {e}")
            print(f"Image shape: {img_uint8.shape}, dtype: {img_uint8.dtype}, min: {img_uint8.min()}, max: {img_uint8.max()}")