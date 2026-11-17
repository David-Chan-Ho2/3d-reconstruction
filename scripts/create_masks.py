import argparse
import io
from pathlib import Path

from PIL import Image
from rembg import remove


def main():
    parser = argparse.ArgumentParser(description="Generate foreground masks for images using rembg.")
    parser.add_argument("image_dir", type=Path, help="Directory of input .JPG images")
    parser.add_argument("mask_dir", type=Path, help="Directory to write output masks to")
    args = parser.parse_args()

    args.mask_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.image_dir.glob("*.JPG"):

        with open(image_path, "rb") as f:
            output = remove(f.read())

        result = Image.open(io.BytesIO(output)).convert("RGBA")

        # Alpha channel becomes the mask
        mask = result.getchannel("A")

        # Force binary black/white mask
        mask = mask.point(lambda x: 255 if x > 128 else 0)

        output_path = args.mask_dir / (image_path.name + ".png")
        mask.save(output_path)

        print(f"Created {output_path}")


if __name__ == "__main__":
    main()