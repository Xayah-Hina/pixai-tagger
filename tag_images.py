import argparse
from pathlib import Path
import sys

from imgutils.tagging import get_pixai_tags

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(
        description="PixAI Tagger v0.9 - one txt per image"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--general-threshold",
        type=float,
        default=0.3,
        help="General tag threshold (model card default: 0.3)",
    )
    parser.add_argument(
        "--character-threshold",
        type=float,
        default=0.85,
        help="Character tag threshold (model card default: 0.85)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    if not input_dir.is_dir():
        print(f"ERROR: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not images:
        print("WARNING: no images found", file=sys.stderr)

    for img_path in images:
        try:
            general, character = get_pixai_tags(
                str(img_path),
                model_name="v0.9",
                fmt=("general", "character"),
            )

            general_tags = [
                k for k, v in general.items()
                if float(v) >= args.general_threshold
            ]

            character_tags = [
                k for k, v in character.items()
                if float(v) >= args.character_threshold
            ]

            tags = general_tags + character_tags

            out_path = img_path.with_suffix(".txt")
            with out_path.open("w", encoding="utf-8") as f:
                f.write(", ".join(tags))

            print(f"OK: {img_path.name} → {out_path.name}")

        except Exception as e:
            print(f"ERROR: {img_path.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
