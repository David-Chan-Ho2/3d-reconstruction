import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Downscale .JPG images to thumbnails, preserving aspect ratio.")
    parser.add_argument("input_dir", type=Path, help="Directory of input .JPG images")
    parser.add_argument("output_dir", type=Path, help="Directory to write resized images to")
    parser.add_argument("--max-size", type=int, default=600, help="Maximum width/height in pixels (default: 600)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.input_dir.glob("*.JPG"):
        img = Image.open(path)

        # Keep aspect ratio, maximum width/height
        img.thumbnail((args.max_size, args.max_size))

        output = args.output_dir / path.name

        img.save(
            output,
            "JPEG",
            quality=85,
            optimize=True
        )

        print(path.name, "->", output)


if __name__ == "__main__":
    main()