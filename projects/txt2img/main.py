import sys
from pathlib import Path

# ---------- 正确的工程根目录（不会错） ----------
# 自动获取当前文件所在目录为基准
CURRENT_FILE = Path(__file__).resolve()
# 往上找 3 级不一定对，我改成 **自动找项目根**（最稳）
PROJECT_ROOT = CURRENT_FILE.parent
while not (PROJECT_ROOT / "projects").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        break  # 防止死循环

sys.path.insert(0, str(PROJECT_ROOT))

# 现在可以安全导入
try:
    from projects.txt2img.generate import generate_image
except Exception as e:
    print("❌ 导入失败，请检查目录结构")
    print("错误：", e)
    exit()

prompt = "masterpiece, best quality, ultra-detailed, 1girl, cute gentle anime girl, soft face, big shiny warm eyes, gentle smile, little blush, long soft brown hair, kawaii aesthetic, clean white background, soft pastel lighting, beautiful detailed eyes, delicate anime style"
negative_prompt = "worst quality, low quality, ugly, deformed, blurry, bad anatomy, bad hands, extra fingers, mutated, disfigured"
output_path = PROJECT_ROOT / "outputs/Anime Girl new1.8.png"

output_path.parent.mkdir(parents=True, exist_ok=True)

# 生成
try:
    generate_image(prompt, output_path, negative_prompt=negative_prompt)
    print("✅ 图片生成完成:", output_path)
except Exception as e:
    print("❌ 生成失败：", e)