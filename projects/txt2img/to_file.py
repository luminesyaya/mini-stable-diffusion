import torch
from safetensors.torch import load_file
from pathlib import Path

# ===== 路径配置（不要改错）=====
safetensors_path = Path(
    "/home/featurize/work/stable-diffusion/models/stable-diffusion-v1-5/loras/Anime_Dragon_Girl.safetensors"
)

output_dir = Path(
    "/home/featurize/work/stable-diffusion/models/stable-diffusion-v1-5/loras/Anime_Dragon_Girl"
)

# ===== 开始转换 =====
output_dir.mkdir(parents=True, exist_ok=True)

print("📦 正在读取 safetensors ...")
state_dict = load_file(str(safetensors_path))

bin_path = output_dir / "pytorch_lora_weights.bin"

print("💾 正在写入 bin ...")
torch.save(state_dict, bin_path)

print(f"✅ 转换完成：{bin_path}")