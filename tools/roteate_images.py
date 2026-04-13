"""
fix_exif_orientation.py

Applies PIL's exif_transpose to every image in a directory, correcting
the physical pixel layout to match the EXIF-displayed orientation.
Useful when COCO annotations were written based on the displayed orientation
but the raw pixel data is stored rotated.

Usage:
    # Overwrite in-place:
    python fix_exif_orientation.py --input_dir /path/to/images

    # Save to a separate output directory:
    python fix_exif_orientation.py --input_dir /path/to/images --output_dir /path/to/output

    # Dry run (no files written):
    python fix_exif_orientation.py --input_dir /path/to/images --dry_run
"""

import os
import argparse
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def fix_exif_orientation(input_dir: str, output_dir: str | None, dry_run: bool):
    input_dir = os.path.abspath(input_dir)

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print("Mode: in-place overwrite")

    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print("No supported image files found.")
        return

    print(f"Found {len(files)} image(s). dry_run={dry_run}\n")

    skipped = 0
    converted = 0

    for fname in sorted(files):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname) if output_dir else src_path

        with Image.open(src_path) as img:
            original_size = img.size
            transposed = ImageOps.exif_transpose(img)
            new_size = transposed.size

        if original_size == new_size:
            print(f"  [skip]    {fname}  {original_size} — no EXIF rotation needed")
            skipped += 1
            continue

        print(f"  [convert] {fname}  {original_size} -> {new_size}")
        converted += 1

        if not dry_run:
            # Re-open to save (exif_transpose returns a copy without EXIF orientation tag)
            with Image.open(src_path) as img:
                transposed = ImageOps.exif_transpose(img)
                # Preserve original format where possible
                fmt = img.format or 'PNG'
                save_kwargs = {}
                if fmt in ('JPEG', 'JPG'):
                    save_kwargs['quality'] = 95
                    save_kwargs['subsampling'] = 0
                transposed.save(dst_path, format=fmt, **save_kwargs)

    print(f"\nDone. {converted} converted, {skipped} skipped.")
    if dry_run:
        print("(dry run — no files were written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix EXIF orientation for all images in a directory.")
    parser.add_argument('--input_dir',  required=True,  help="Directory containing input images")
    parser.add_argument('--output_dir', default=None,   help="Output directory (default: overwrite in-place)")
    parser.add_argument('--dry_run',    action='store_true', help="Print what would be done without writing files")
    args = parser.parse_args()

    fix_exif_orientation(args.input_dir, args.output_dir, args.dry_run)