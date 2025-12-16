FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install PyTorch CUDA 12.8
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# PixAI tagger deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tag_images.py .

ENTRYPOINT ["python", "/app/tag_images.py"]
