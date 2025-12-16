FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# 🔑 Hugging Face cache location (important)
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# --------------------------------------------------
# Install PyTorch (CUDA 12.8 wheels)
# --------------------------------------------------
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# --------------------------------------------------
# PixAI tagger deps
# --------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# 🔥 Bake PixAI Tagger model into image (v0.9)
# --------------------------------------------------
RUN python - << 'EOF'
from huggingface_hub import hf_hub_download

repo = "deepghs/pixai-tagger-v0.9-onnx"
files = [
    "model.onnx",
    "selected_tags.csv",
    "preprocess.json",
    "thresholds.csv",
]

for f in files:
    try:
        path = hf_hub_download(repo_id=repo, filename=f)
        print("Cached:", path)
    except Exception as e:
        print("Skip:", f, e)
EOF

# --------------------------------------------------
# Your app
# --------------------------------------------------
COPY tag_images.py .

ENTRYPOINT ["python", "/app/tag_images.py"]
