import sys
from pathlib import Path

# ---------- 正确的工程根目录 ----------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
while not (PROJECT_ROOT / "projects").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        break

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from projects.txt2img.generate import generate_image
except Exception as e:
    print("❌ 导入失败，请检查目录结构")
    print("错误：", e)
    exit()

# ---------- LoRA 配置 ----------
LORA_PATH = "/home/featurize/work/stable-diffusion/outputs/lora_iori_q"
TRIGGER_WORD = "iori_q"

# ---------- Prompt ----------
prompt = (
    f"{TRIGGER_WORD}, "
    "masterpiece, best quality, ultra-detailed, "
    "1girl, cute gentle anime girl, soft face, "
    "big shiny warm eyes, gentle smile, little blush, "
    "long soft brown hair, kawaii aesthetic, "
    "clean white background, soft pastel lighting, "
    "beautiful detailed eyes, delicate anime style"
)

negative_prompt = (
    "worst quality, low quality, ugly, deformed, "
    "blurry, bad anatomy, bad hands, extra fingers, "
    "mutated, disfigured"
)

# ---------- 输出 ----------
output_path = PROJECT_ROOT / "outputs/Anime_Girl_newlora1.8.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

# ---------- 生成 ----------
try:
    generate_image(
        prompt,
        output_path,
        negative_prompt=negative_prompt,
        lora_path=LORA_PATH,
    )
    print("✅ 图片生成完成:", output_path)
except Exception as e:
    print("❌ 生成失败：", e)