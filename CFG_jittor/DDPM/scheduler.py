import math
import jittor as jt

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return jt.linspace(beta_start, beta_end, timesteps).to(jt.float64)

def sigmoid_beta_schedule(timesteps):
    betas = jt.linspace(-6, 6, timesteps, dtype=jt.float64)
    betas = jt.sigmoid(betas) / (jt.max(betas) - jt.min(betas)) * (0.02 - jt.min(betas)) / 10
    return betas

def cosine_beta_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = jt.linspace(0, timesteps, steps, dtype=jt.float64)
    alphas_cumprod = jt.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jt.clip(betas, 0, 0.999)