import torch
import logging
from mini_pipeline_00 import MiniPipeline
from config import MODEL_PATH, DEVICE, DTYPE
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pipeline(
    lora_path: str = None,
    lora_weight_name: str = "pytorch_lora_weights.safetensors",
):
    """
    每次都创建一个新的 MiniPipeline，
    避免 LoRA 权重残留问题
    """
    return MiniPipeline(
        model_path=MODEL_PATH,
        lora_path=lora_path,
        lora_weight_name=lora_weight_name,
        device=DEVICE,
        dtype=getattr(torch, DTYPE),
    )


def generate_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    lora_path: str = None,
):
    logger.info(f"Generating image with LoRA: {lora_path}")

    pipeline = get_pipeline(lora_path=lora_path)

    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_steps=50,
        guidance_scale=7.5,
        seed=42,
    )

    image = Image.fromarray((image * 255).astype("uint8"))
    image.save(output_path)

    logger.info(f"Image saved to {output_path}")