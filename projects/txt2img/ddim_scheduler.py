from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class DDIMSchedulerOutput:
    prev_sample:torch.tensor
    pred_original_sample:torch.tensor
    
class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps: int=1000,
        beta_start: float=0.0001,
        beta_end: float=0.02,
        eta: float=0.0,
        clip_sample: bool=True,
        clip_range: float=1.0
    ):
        if beta_start >= beta_end:
            raise ValueError("beta_start must be smaller than beta_end")
        
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta
        self.clip_sample = clip_sample
        self.clip_range = clip_range
        
        self.betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps
        )
        self.alphas = 1.0 - self.betas
        
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.timesteps = None
    
    def set_timesteps(self, num_inference_steps:int):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("Inference steps exceed training steps")
        
        if num_inference_steps <= 0:
            raise ValueError(
            f"[DDIMScheduler] num_inference_steps must be > 0, "
            f"got {num_inference_steps}"
        )
        
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        ).long()
    
    def step(self, model_out: torch.Tensor, timestep: int, sample: torch.Tensor) -> DDIMSchedulerOutput:
        if self.timesteps is None:
            raise RuntimeError("Call set_timesteps() before step()")

        device = sample.device
        dtype = sample.dtype

        # 官方风格：移到正确设备和精度
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=dtype)
        final_alpha_cumprod = self.final_alpha_cumprod.to(device=device, dtype=dtype)

        # --------------------------
        # ✅ 官方核心：直接数学计算 prev_timestep（不查索引！）
        # --------------------------
        step_ratio = self.num_train_timesteps // len(self.timesteps)
        prev_timestep = timestep - step_ratio

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod

        beta_prod_t = 1.0 - alpha_prod_t

        # 预测原图
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_out) / alpha_prod_t.sqrt()

        # 安全裁剪
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_range, self.clip_range)

        # 方差
        variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = self.eta * variance.sqrt()

        # 预测方向
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).sqrt() * model_out

        # 上一步结果
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        # 噪声
        if self.eta > 0:
            noise = torch.randn_like(sample)
            prev_sample += std_dev_t * noise

        # 安全锁
        prev_sample = torch.nan_to_num(prev_sample, nan=0.0)

        return DDIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )