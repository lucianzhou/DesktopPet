#!/usr/bin/env python3
"""Extract a generated flat-key strip into shared-scale transparent cells."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

_helper_path = Path(__file__).resolve().with_name("prepare-canonical.py")
_helper_spec = importlib.util.spec_from_file_location("prepare_canonical", _helper_path)
if _helper_spec is None or _helper_spec.loader is None:
    raise ImportError(f"Unable to load {_helper_path}")
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
CELL_SIZE = _helper.CELL_SIZE
CONTENT_SIZE = _helper.CONTENT_SIZE
BASELINE_Y = _helper.BASELINE_Y
remove_magenta_key = _helper.remove_magenta_key


def keep_main_component(image: Image.Image) -> Image.Image:
    arr = np.asarray(image, dtype=np.uint8).copy()
    mask = arr[..., 3] > 8
    height, width = mask.shape
    seen = np.zeros_like(mask)
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    if not components:
        raise ValueError("slot has no visible component")
    interior = [component for component in components if not any(x == 0 or x == width - 1 for _, x in component)]
    main = max(interior or components, key=len)
    keep = np.zeros_like(mask)
    for y, x in main:
        keep[y, x] = True
    arr[~keep, 3] = 0
    arr[arr[..., 3] == 0, :3] = 0
    return Image.fromarray(arr, "RGBA")


def extract_full_components(image: Image.Image, count: int) -> list[Image.Image]:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    mask = arr[..., 3] > 8
    height, width = mask.shape
    seen = np.zeros_like(mask)
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) > 500:
                components.append(component)
    if len(components) < count:
        raise ValueError(f"found only {len(components)} complete components, expected {count}")
    components = sorted(components, key=len, reverse=True)[:count]
    components.sort(key=lambda points: min(x for _, x in points))
    result: list[Image.Image] = []
    for points in components:
        ys = [y for y, _ in points]
        xs = [x for _, x in points]
        left, top, right, bottom = min(xs), min(ys), max(xs) + 1, max(ys) + 1
        crop = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
        for y, x in points:
            crop[y - top, x - left] = arr[y, x]
        result.append(Image.fromarray(crop, "RGBA"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--shared-scale", type=float)
    parser.add_argument(
        "--already-alpha",
        action="store_true",
        help="Treat the input as an already keyed RGBA image instead of removing magenta.",
    )
    parser.add_argument(
        "--raw-crops",
        action="store_true",
        help="Write ordered complete-component crops without fitting them into 192x208 cells.",
    )
    args = parser.parse_args()

    source = Image.open(args.input).convert("RGBA")
    keyed = source if args.already_alpha else remove_magenta_key(source)
    crops = extract_full_components(keyed, args.count)

    if args.raw_crops:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, crop in enumerate(crops):
            crop.save(args.output_dir / f"{index:02d}.png")
        print({"count": args.count, "raw_crops": True})
        return

    derived_scale = min(
        CONTENT_SIZE[0] / max(crop.width for crop in crops),
        CONTENT_SIZE[1] / max(crop.height for crop in crops),
    )
    shared_scale = args.shared_scale if args.shared_scale is not None else derived_scale
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, crop in enumerate(crops):
        size = (max(1, round(crop.width * shared_scale)), max(1, round(crop.height * shared_scale)))
        resized = crop.resize(size, Image.Resampling.LANCZOS)
        cell = Image.new("RGBA", CELL_SIZE, (0, 0, 0, 0))
        x = (CELL_SIZE[0] - size[0]) // 2
        y = BASELINE_Y - size[1]
        cell.alpha_composite(resized, (x, y))
        arr = np.asarray(cell, dtype=np.uint8).copy()
        arr[arr[..., 3] == 0, :3] = 0
        Image.fromarray(arr, "RGBA").save(args.output_dir / f"{index:02d}.png")

    print({"count": args.count, "shared_scale": shared_scale})


if __name__ == "__main__":
    main()
