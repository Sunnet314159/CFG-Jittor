import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

from DDPM.unet import UnetModel
from DDPM.ddpm import GaussianDiffusion

from utils.generate import generate_images
from utils.plot import plot_loss

from utils.save import save_model, load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size=128):
    transform_ops = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # [-1,1]
    ])
    dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform_ops)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, gaussian_diffusion, train_loader, epochs=10, timesteps=500, p_uncond=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    len_data = len(train_loader)
    time_end = time.time()

    for epoch in range(epochs):
        for step, (images, labels) in enumerate(train_loader):
            time_start = time_end
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            batch_size = images.size(0)
            z_uncond = torch.rand(batch_size, device=device)
            batch_mask = (z_uncond > p_uncond).long()

            t = torch.randint(0, timesteps, (batch_size,), dtype=torch.long, device=device)

            loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)

            if step % 100 == 0:
                time_end = time.time()
                log_message = (f"Epoch {epoch+1}/{epochs}\t Step {step+1}/{len_data}\t "
                               f"Loss {loss.item():.4f}\t Time {time_end-time_start:.2f}")
                logging.info(log_message)

            loss.backward()
            optimizer.step()

def main():
    log_file_path = 'training_progress.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info("training started.")

    device_message = f"PyTorch using {'CUDA' if torch.cuda.is_available() else 'CPU'}."
    logging.info(device_message)

    batch_size = 128
    timesteps = 500
    image_size = 28
    channels = 1
    n_class = 10
    epochs = 50
    p_uncond = 0.2
    w = 2

    train_loader = load_data(batch_size=batch_size)

    model = UnetModel(
        in_channels=1,
        model_channels=96,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[],
        class_num=n_class
    ).to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule='linear')

    train_model(model, gaussian_diffusion, train_loader, epochs=epochs, timesteps=timesteps, p_uncond=p_uncond)
    logging.info("training finished.")

    save_model(model, path='./saved_models/cfg_ddpm.pth')

    model = load_model(path='./saved_models/cfg_ddpm.pth', model=model).to(device)

    fid_score, is_score = generate_images(
        gaussian_diffusion, model, train_loader, batch_size=64, channels=channels, image_size=image_size,
        n_class=n_class, w=w
    )
    logging.info(f"FID: {fid_score:.2f}, IS: {is_score[0]:.2f} (std: {is_score[1]:.2f})")

    plot_loss(log_filepath="training_progress.log", output_image_filepath="training_loss_plot.png")

if __name__ == "__main__":
    main()
