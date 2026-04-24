import torch
import logging
from mini_pipeline_00 import MiniPipeline
from config import MODEL_PATH, DEVICE, DTYPE
from PIL import Image

_pipeline = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = MiniPipeline(
            model_path=MODEL_PATH,
            device=DEVICE,
            dtype=getattr(torch, DTYPE)
        )
    return _pipeline

def generate_image(prompt:str, output_path:str, negative_prompt: str = ""): # , lora_path: str = None, lora_scale: float = 0.75
    pipeline = get_pipeline()
    
    # if lora_path is not None:
    #     logger.info(f"Using LoRA: {lora_path}, scale={lora_scale}")
    #     pipeline.load_lora(lora_path, scale=lora_scale)
    
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_steps=50,
        guidance_scale=7.5,
        seed=42
    )
    image = Image.fromarray((image * 255).astype('uint8'))
    image.save(output_path)
    
    # if lora_path is not None:
    #     pipeline.unload_lora()