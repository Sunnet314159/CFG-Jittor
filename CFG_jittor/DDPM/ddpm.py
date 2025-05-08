import numpy as np
import jittor as jt
from jittor import nn
from tqdm import tqdm

from .scheduler import linear_beta_schedule, sigmoid_beta_schedule, cosine_beta_schedule


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = jt.cumprod(self.alphas)
        self.alphas_cumprod_prev = jt.nn.pad(self.alphas_cumprod[:-1], [1, 0], value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = jt.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jt.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jt.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jt.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jt.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = jt.log(
            jt.concat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * jt.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * jt.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward：q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jt.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # obtain mean and variance of q(x_t | x_0) 
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # calculate posterior mean and variance of q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # calculate x_0 form x_t and noise
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # calculate the predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, c, w, clip_denoised=True):
        batch_size = x_t.shape[0]
        pred_noise_c = model(x_t, t, c, jt.ones([batch_size], dtype=jt.int))
        pred_noise_none = model(x_t, t, c, jt.zeros([batch_size], dtype=jt.int))
        pred_noise = (1+w)*pred_noise_c - w*pred_noise_none
        
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = jt.clamp(x_recon, -1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    
    # use x_t and noise to obtain  x_{t-1}
    @jt.no_grad()
    def p_sample(self, model, x_t, t, c, w, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                c, w, clip_denoised=clip_denoised)
        noise = jt.randn_like(x_t)
        nonzero_mask = ((t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1))))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # reverse: DDPM sampling
    @jt.no_grad()
    def ddpm_sample(self, model, shape, n_class=10, w=2, mode='random', clip_denoised=True):
        batch_size = shape[0]

        if mode == 'random':
            cur_y = jt.randint(0, n_class, (batch_size,))
        elif mode == 'all':
            if batch_size % n_class != 0:
                batch_size = n_class
                print('change batch_size to', n_class)
            cur_y = jt.array([x for x in range(n_class)]*(batch_size//n_class), dtype=jt.int)
        else:
            cur_y = jt.ones([batch_size], dtype=jt.int) * int(mode)

        img = jt.randn(shape)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='DDPM sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, jt.full((batch_size,), i, dtype=jt.int), cur_y, w, clip_denoised)
            imgs.append(img.numpy())
        return imgs
    
    # sampling function for the model
    @jt.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=1, n_class=10, w=2, mode='random', clip_denoised=True):
        return self.ddpm_sample(model, (batch_size, channels, image_size, image_size), n_class, w, mode, clip_denoised)
    
    # calculate the loss function
    def train_losses(self, model, x_start, t, c, mask_c):
        noise = jt.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, c, mask_c)
        loss = nn.mse_loss(noise, predicted_noise)
        return loss