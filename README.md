这份 README 不仅整合了你现有的步骤，还结合了我们在调试过程中遇到的问题（如 huggingface-hub 版本、safetensors 提示等），使其更加健壮。

Stable Diffusion 本地化部署与自定义 Pipeline 实践

本项目旨在构建一个脱离 HuggingFace Hub 直连、完全本地化的 Stable Diffusion 推理环境，并实现自定义的轻量级推理 Pipeline，深入理解 Diffusers 库的内部工作机制。

一、项目结构

项目采用模块化设计，将源码、模型与业务代码分离：
stable-diffusion/
├── diffusers/              # diffusers 源码 (Git 克隆)
├── models/
│   └── stable-diffusion-v1-5/  # 本地模型权重
├── projects/
│   └── txt2img/
│       ├── main.py        # 程序入口
│       ├── generate.py    # 生成逻辑封装
│       └── mini_pipeline.py # 自定义 MiniPipeline 实现
└── README.md              # 本文档


二、环境准备

1. 创建并进入项目目录

mkdir -p stable-diffusion
cd stable-diffusion


2. 克隆 Diffusers 库

我们将使用源码安装，以便进行深度定制和理解内部机制。
git clone https://github.com/huggingface/diffusers.git


3. 创建并激活虚拟环境

推荐使用 venv 进行环境隔离：
python -m venv diffusers-envs
source diffusers-envs/bin/activate


激活成功后，终端前缀将变为 (diffusers-envs)。

三、安装依赖

1. 安装 PyTorch (GPU 推荐)

根据你的 CUDA 版本选择合适的 PyTorch。以下以 CUDA 12.1 为例（参考链接1中的 cu121 源）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


注：你也可以直接使用 diffusers 的捆绑版本：pip install --upgrade "diffusers[torch]"

2. 以可编辑模式安装 Diffusers

进入源码目录并进行开发模式安装，这样你对源码的修改会立即生效：
cd diffusers
pip install -e .
pip install -e .[dev]  # 安装开发依赖（如 pytest, black 等）
cd ..


3. 修复与锁定依赖版本

根据我们的调试经验，需要明确指定 huggingface-hub 的版本以避免 API 不兼容问题（如 DDUFEntry 错误）：
pip install "huggingface-hub>=0.21.0,<1.0"
pip install -U transformers


四、模型下载（防 429 稳定方案）

为了避免直接从 HuggingFace Hub 下载模型时的网络波动和速率限制，我们采用 hf download 命令行工具进行离线下载。

1. 安装 HuggingFace Hub 工具

pip install huggingface-hub


2. 创建模型目录

mkdir -p models/stable-diffusion-v1-5


3. 下载模型

使用 --local-dir 指定本地路径，并通过 --max-workers 1 降低并发以减少被限流的风险：
huggingface-cli download \
  stabilityai/stable-diffusion-v1-5 \
  --local-dir models/stable-diffusion-v1-5 \
  --local-dir-use-symlinks False \
  --max-workers 1


提示：如果遇到 safetensors 相关的警告，属于正常现象，不影响模型功能。若想消除警告，可手动下载 diffusion_pytorch_model.safetensors 文件放入模型目录。

4. 验证模型完整性

du -sh models/stable-diffusion-v1-5


正常情况下应显示数 GB 的大小。

五、运行项目

确保所有环境准备就绪后，回到项目根目录运行：
(diffusers-envs) ➜ stable-diffusion python projects/txt2img/main.py


如果一切顺利，你将看到类似以下的输出，并在 outputs/ 目录下生成图片：
INFO:Initializing Mini Stable Diffusion Pipeline...
INFO:Loading all models...
INFO:Starting image generation with prompt: a squirrel in Picasso style
INFO:Image generated successfully!
✅ 图片生成完成: /home/featurize/work/stable-diffusion/outputs/squirrel.png

基于 🤗 Diffusers 库构建，遵循其模块化与可用性的设计哲学。