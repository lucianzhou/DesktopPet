#!/usr/bin/env python3
"""Prepare the approved Baomihua master as a clean, registered sprite cell."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


CELL_SIZE = (192, 208)
CONTENT_SIZE = (176, 196)
BASELINE_Y = 204


def remove_magenta_key(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[..., :3].astype(np.int16)
    source_alpha = rgba[..., 3].astype(np.float32) / 255.0

    # The generated background varies in brightness but remains strongly
    # magenta: both red and blue are much higher than green. Cream/white fur
    # has a small score and is therefore preserved.
    score = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
    matte = np.ones(score.shape, dtype=np.float32)
    matte[score >= 75] = 0.0
    transition = (score > 35) & (score < 75)
    matte[transition] = (75.0 - score[transition]) / 40.0
    alpha = np.rint(255.0 * matte * source_alpha).astype(np.uint8)

    # Replace antialiased edge RGB with the nearest opaque interior color.
    # This preserves alpha while preventing magenta hidden RGB from becoming
    # a halo during high-quality scaling and AppKit compositing.
    colors = rgba[..., :3].copy()
    known = alpha >= 250
    edge_band = np.zeros_like(known)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        shifted = np.roll(np.roll(alpha < 250, dy, axis=0), dx, axis=1)
        edge_band |= shifted
    # Opaque pink/magenta fringe pixels are only corrected when they sit on
    # the silhouette boundary; interior warm pink (notably the nose) remains
    # untouched.
    fringe = edge_band & (score > 25) & (alpha > 0)
    known &= ~fringe
    pending = ((alpha > 0) & ~known) | fringe
    height, width = alpha.shape
    queue: deque[tuple[int, int]] = deque(map(tuple, np.argwhere(known)))
    visited = known.copy()
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if visited[ny, nx] or not pending[ny, nx]:
                continue
            colors[ny, nx] = colors[y, x]
            visited[ny, nx] = True
            queue.append((ny, nx))

    colors[alpha == 0] = 0
    output = np.dstack((colors, alpha))
    return Image.fromarray(output, "RGBA")


def register_cell(image: Image.Image) -> tuple[Image.Image, dict[str, object]]:
    rgba = np.asarray(image, dtype=np.uint8)
    visible = rgba[..., 3] > 8
    ys, xs = np.nonzero(visible)
    if len(xs) == 0:
        raise ValueError("No visible sprite pixels after chroma removal")

    source_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crop = image.crop(source_box)
    scale = min(CONTENT_SIZE[0] / crop.width, CONTENT_SIZE[1] / crop.height)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    resized = crop.resize(size, Image.Resampling.LANCZOS)

    cell = Image.new("RGBA", CELL_SIZE, (0, 0, 0, 0))
    x = (CELL_SIZE[0] - size[0]) // 2
    y = BASELINE_Y - size[1]
    cell.alpha_composite(resized, (x, y))

    cell_array = np.asarray(cell, dtype=np.uint8).copy()
    cell_array[cell_array[..., 3] == 0, :3] = 0
    cell = Image.fromarray(cell_array, "RGBA")
    report = {
        "cell_size": list(CELL_SIZE),
        "content_limit": list(CONTENT_SIZE),
        "source_box": list(source_box),
        "registered_box": [x, y, x + size[0], y + size[1]],
        "baseline_y": BASELINE_Y,
        "scale": scale,
        "transparent_corners": all(cell.getpixel(point)[3] == 0 for point in ((0, 0), (191, 0), (0, 207), (191, 207))),
    }
    return cell, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--alpha-output", required=True, type=Path)
    parser.add_argument("--cell-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    keyed = remove_magenta_key(Image.open(args.input))
    cell, report = register_cell(keyed)
    args.alpha_output.parent.mkdir(parents=True, exist_ok=True)
    args.cell_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    keyed.save(args.alpha_output)
    cell.save(args.cell_output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
