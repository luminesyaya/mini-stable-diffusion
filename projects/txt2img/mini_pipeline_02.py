import sys
import torch
import logging
from pathlib import Path

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel
from config import MODEL_PATH, DEVICE, DTYPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MiniPipeline:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
    ):
        logger.info("Initializing Mini Stable Diffusion Pipeline ...")

        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        self._lora_loaded = False
        self._lora_scale = 1.0

        self._load_models()

    # ========================= 模型加载 =========================
    def _load_models(self):
        logger.info("Loading all models ...")

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

        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )

        logger.info("All models loaded successfully.")

    def load_lora(self, lora_path: str, scale: float = 1.0):
        try:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                raise FileNotFoundError(lora_path)

            # ✅ PEFT 标准文件名
            safetensors_file = lora_path / "adapter_model.safetensors"
            if not safetensors_file.exists():
                raise FileNotFoundError(safetensors_file)

            # ✅ 读取 safetensors（不关心 diffusers 认不认识）
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_file))

            # ✅ 手动注入到 UNet（0.26.x 唯一稳的方式）
            default_attn_processor = {
                name: AttnProcessor()
                for name in self.unet.attn_processors.keys()
            }

            self.unet.set_attn_processor(default_attn_processor)
            self.unet.load_state_dict(state_dict, strict=False)

            self._lora_loaded = True
            self._lora_scale = scale

            logger.info(f"✅ LoRA loaded: {safetensors_file}, scale={scale}")

        except Exception as e:
            logger.warning(f"LoRA 失败: {e}", exc_info=True)
            self._lora_loaded = False

    def unload_lora(self):
        try:
            # 重新设置为默认 processor
            default_attn_processor = {
                name: AttnProcessor()
                for name in self.unet.attn_processors.keys()
            }
            self.unet.set_attn_processor(default_attn_processor)
            self._lora_loaded = False
            logger.info("✅ LoRA unloaded")
        except Exception:
            pass

    # ========================= 文本编码 =========================
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        neg_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            pos_emb = self.text_encoder(tokens)[0]
            neg_emb = self.text_encoder(neg_tokens)[0]

        return torch.cat([neg_emb, pos_emb])

    # ========================= latent 初始化 =========================
    def init_latents(self, batch_size=1, height=512, width=512, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        return torch.randn(
            (batch_size, 4, height // 8, width // 8),
            device=self.device,
            dtype=self.dtype,
        )

    # ========================= ✅ UNet 去噪（LoRA scale 生效点） =========================
    def denoise_step(self, latents, timestep, text_embeddings, guidance_scale=7.5):
        latent_model_input = torch.cat([latents] * 2)

        cross_attn_kwargs = {}
        if self._lora_loaded:
            cross_attn_kwargs["scale"] = self._lora_scale

        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attn_kwargs,
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # ========================= VAE 解码 =========================
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        return image.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

    # ========================= Pipeline 主入口 =========================
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
    ):
        logger.info(f"Starting image generation with prompt: {prompt}")

        text_embeddings = self.encode_prompt(prompt, negative_prompt)
        latents = self.init_latents(height=height, width=width, seed=seed)

        self.scheduler.set_timesteps(num_steps)

        for t in self.scheduler.timesteps:
            latents = self.denoise_step(
                latents, t, text_embeddings, guidance_scale
            )

        image = self.decode_latents(latents)
        logger.info("Image generated successfully!")
        return image