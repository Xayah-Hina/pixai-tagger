# PixAI Tagger (Docker)

Image tagging tool based on PixAI Tagger v0.9.

## Requirements
- Docker
- NVIDIA GPU + driver (for CUDA version)

## Usage
```bash
docker run --rm --gpus all \
  -v /path/to/images:/data/images \
  ghcr.io/<your-username>/pixai-tagger:latest \
  --input /data/images

