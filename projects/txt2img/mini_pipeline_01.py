# 加载模型
import sys
import torch
import logging
from pathlib import Path
from typing import Optional
DIFFUSERS_ROOT = Path("/home/featurize/work/stable-diffusion/diffusers")
sys.path.insert(0, str(DIFFUSERS_ROOT / "src"))


from diffusers import UNet2DConditionModel, AutoencoderKL
from projects.txt2img.ddim_scheduler import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from projects.txt2img.config import MODEL_PATH, DEVICE, DTYPE

import projects.txt2img.ddim_scheduler as ddim
print(">>> ACTUAL DDIM SCHEDULER PATH:", ddim.__file__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 对外暴露统一接口
class MiniPipeline:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE 
        ):
        logger.info("Initializing Mini Stable Diffusion Pipeline...")
        
        # 统一设备和精度管理
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        self._load_models()
        
    def _load_models(self):
        logger.info("Loading all models...")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
        self.model_path, subfolder="tokenizer"
    )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
        self.model_path, subfolder="text_encoder"
    ).to(self.device, self.dtype)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_path, subfolder="unet"
        ).to(self.device, self.dtype)
        
        self.vae = AutoencoderKL.from_pretrained(
            self.model_path, subfolder="vae"
        ).to(self.device, self.dtype)
        
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            eta=0.0,              
            clip_sample=True,
        )
        
        logger.info("All models loaded successfully.")
        
    # 编码文本
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty.")
        
        tokens = self.tokenizer(
            prompt,
            padding = "max_length",
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt"
        ).input_ids.to(DEVICE)
        
        neg_tokens = self.tokenizer(
            negative_prompt,
            padding = "max_length",
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt"
        ).input_ids.to(DEVICE)
        
        with torch.no_grad():
            pos_emb = self.text_encoder(tokens)[0]
            neg_emb = self.text_encoder(neg_tokens)[0]
            
        pos_emb = pos_emb.to(self.dtype)
        neg_emb = neg_emb.to(self.dtype)
        
        text_embeddings = torch.cat([neg_emb, pos_emb])
        
        logger.info(f"Text embedding shape: {text_embeddings.shape}")
        return text_embeddings
    
    # 初始化 latent
    def init_latents(self, batch_size: int = 1, height: int = 512, width: int = 512, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
            
        latents = torch.randn(
            (batch_size, 4, height//8, width//8),
            device = self.device,
            dtype = self.dtype
        )
        
        return latents
    
    # VAE 解码
    def decode_latents(self, latents: torch.Tensor):
        latents = latents / self.vae.config.scaling_factor
        
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image/2 + 0.5).clamp(0,1)
        image = image.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

        return image
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 512,
        width: int = 512
    ):
        try:
            logger.info(f"Starting image generation with prompt: {prompt}")
            
            # 1.文本编码
            text_embeddings = self.encode_prompt(prompt, negative_prompt)
            
            # 2.初始化噪声
            latents = self.init_latents(height=height, width=width, seed=seed)
            
            # 3.调度器时间步设置
            self.scheduler.set_timesteps(num_steps)

            # 4.循环去噪
            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 2)
                    t_for_model = t.to(self.dtype)
                    noise_pred = self.unet(
                        latent_model_input,
                        t_for_model,
                        encoder_hidden_states = text_embeddings
                    ).sample
                    
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                latents = self.scheduler.step(
                    noise_pred, t, latents
                ).prev_sample
            
            # 5.解码图像   
            image = self.decode_latents(latents)
            
            logger.info("Image generated successfully!")
            return image
        
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise