#!/usr/bin/env python3
"""Extract ordered RGBA cells from a fixed-size sprite atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-width", type=int, default=192)
    parser.add_argument("--cell-height", type=int, default=208)
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGBA")
    if args.cell_width <= 0 or args.cell_height <= 0 or args.columns <= 0:
        raise ValueError("cell dimensions and columns must be positive")
    if image.width % args.cell_width or image.height % args.cell_height:
        raise ValueError(
            f"atlas size {image.size} is not divisible by "
            f"{args.cell_width}x{args.cell_height}"
        )
    actual_columns = image.width // args.cell_width
    if actual_columns != args.columns:
        raise ValueError(f"atlas has {actual_columns} columns, expected {args.columns}")

    rows = image.height // args.cell_height
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.columns * rows):
        row, column = divmod(index, args.columns)
        bounds = (
            column * args.cell_width,
            row * args.cell_height,
            (column + 1) * args.cell_width,
            (row + 1) * args.cell_height,
        )
        image.crop(bounds).save(args.output_dir / f"{index:02d}.png")

    print({"count": args.columns * rows, "rows": rows, "columns": args.columns})


if __name__ == "__main__":
    main()
