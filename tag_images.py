from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from tqdm import tqdm
from imgutils.tagging import get_pixai_tags

import os


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

BLACKLIST_EXACT = {
    "patreon_username", "fanbox_username",
    "web_address", "signature", "artist_name",
    "watermark", "logo",
}

BLACKLIST_CONTAINS = {
    "logo", "username", "watermark",
}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def build_caption(
    img: Path,
    *,
    general_threshold: float,
    character_threshold: float,
    trigger: str | None,
    feature_mode: bool,
) -> str:
    general, character, *_ = get_pixai_tags(
        str(img),
        model_name="v0.9",
        fmt=("general", "character", "ips", "ips_mapping"),
    )

    tokens: list[str] = []
    seen: set[str] = set()

    def add(tok: str) -> None:
        tok = tok.strip()
        if not tok:
            return
        if tok in seen:
            return
        seen.add(tok)
        tokens.append(tok)

    if trigger:
        add(trigger)

    if not feature_mode and character:
        try:
            best_tag, best_score = max(character.items(), key=lambda x: x[1])
        except Exception:
            best_tag, best_score = None, None

        if isinstance(best_tag, str) and isinstance(best_score, (int, float)) and best_score >= character_threshold:
            add(best_tag)

    for tag, score in getattr(general, "items", lambda: [])():
        if not isinstance(tag, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        if score < general_threshold:
            continue

        tag = tag.strip()
        if not tag:
            continue

        if tag in BLACKLIST_EXACT:
            continue

        lowered = tag.casefold()
        if any(bad in lowered for bad in BLACKLIST_CONTAINS):
            continue

        if feature_mode and ("(" in tag and ")" in tag):
            continue

        add(tag)

    return ", ".join(tokens)



def process_one(
    img: Path,
    *,
    general_threshold: float,
    character_threshold: float,
    trigger: str | None,
    feature_mode: bool,
    skip_existing: bool,
) -> None:
    out_txt = img.with_suffix(".txt")

    if skip_existing and out_txt.exists():
        return

    caption = build_caption(
        img,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        trigger=trigger,
        feature_mode=feature_mode,
    )

    out_txt.write_text(caption, encoding="utf-8")


def iter_images(dir_: Path) -> Iterable[Path]:
    # os.scandir is faster than Path.iterdir for large directories
    with os.scandir(dir_) as it:
        for entry in it:
            if entry.is_file():
                p = Path(entry.path)
                if p.suffix.lower() in IMAGE_EXTS:
                    yield p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PixAI Tagger v0.9 (modern, multithreaded, LoRA-ready)"
    )
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--general-threshold", type=float, default=0.35)
    ap.add_argument("--character-threshold", type=float, default=0.85)
    ap.add_argument("--trigger", type=str, default=None)
    ap.add_argument("--feature-mode", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(2, (os.cpu_count() or 8) - 2),
        help="Recommended: GPU=2–6, CPU=cores-2",
    )

    args = ap.parse_args()

    input_dir = args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        print("No images found.")
        return

    print(
        f"Processing {len(images)} images "
        f"with {args.workers} threads"
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            tqdm(
                pool.map(
                    lambda p: process_one(
                        p,
                        general_threshold=args.general_threshold,
                        character_threshold=args.character_threshold,
                        trigger=args.trigger,
                        feature_mode=args.feature_mode,
                        skip_existing=args.skip_existing,
                    ),
                    images,
                ),
                total=len(images),
                ncols=100,
            )
        )


if __name__ == "__main__":
    main()
