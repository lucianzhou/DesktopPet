#!/usr/bin/env python3
"""Deterministic structural gate for the registered wake atlas."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


CELL_W, CELL_H = 192, 208
BASELINE_Y = 204
CONTENT_X = (8, 184)
MIN_COMPONENT_AREA = 250


def enclosed_transparent(mask: np.ndarray) -> int:
    background = ~mask
    seen = np.zeros_like(background)
    height, width = background.shape
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if background[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if background[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and background[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return int((background & ~seen).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--first-anchor", type=Path)
    parser.add_argument("--last-anchor", type=Path)
    args = parser.parse_args()

    if args.columns <= 0 or args.rows <= 0:
        raise SystemExit("--columns and --rows must both be positive")
    frame_count = args.columns * args.rows

    image = Image.open(args.atlas).convert("RGBA")
    failures: list[str] = []
    frames: list[dict[str, object]] = []
    expected_size = (CELL_W * args.columns, CELL_H * args.rows)
    if image.size != expected_size:
        failures.append(f"atlas size is {image.size}, expected {expected_size}")

    for index in range(frame_count):
        row, column = divmod(index, args.columns)
        cell = image.crop((column * CELL_W, row * CELL_H, (column + 1) * CELL_W, (row + 1) * CELL_H))
        rgba = np.asarray(cell, dtype=np.uint8)
        alpha = rgba[..., 3]
        visible = alpha > 8
        ys, xs = np.where(visible)
        frame_failures: list[str] = []
        if len(xs) == 0:
            frame_failures.append("blank")
            frames.append({"frame": index, "ok": False, "failures": frame_failures})
            failures.append(f"frame {index}: blank")
            continue

        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if left < CONTENT_X[0] or right > CONTENT_X[1]:
            frame_failures.append(f"horizontal bounds {(left, right)} outside {CONTENT_X}")
        if top < 8:
            frame_failures.append(f"top bound {top} is too close to cell edge")
        if bottom != BASELINE_Y:
            frame_failures.append(f"baseline {bottom} != {BASELINE_Y}")
        if alpha[:, 0].max() > 0 or alpha[:, -1].max() > 0 or alpha[0, :].max() > 0 or alpha[-1, :].max() > 0:
            frame_failures.append("visible pixels touch cell edge")

        # Require one substantial connected sprite component.
        seen = np.zeros_like(visible)
        components: list[int] = []
        for y, x in zip(*np.where(visible)):
            if seen[y, x]:
                continue
            queue = [(int(y), int(x))]
            seen[y, x] = True
            size = 0
            while queue:
                cy, cx = queue.pop()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < CELL_H and 0 <= nx < CELL_W and visible[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            components.append(size)
        substantial = [size for size in components if size >= MIN_COMPONENT_AREA]
        if len(substantial) != 1:
            frame_failures.append(f"expected one substantial component, found {len(substantial)}")

        if enclosed_transparent(visible[top:bottom, left:right]) > 16:
            frame_failures.append("enclosed transparent hole inside sprite bounds")

        rgb = rgba[..., :3].astype(np.int16)
        score = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
        if int(((alpha > 0) & (alpha < 250) & (score > 35)).sum()) > 180:
            frame_failures.append("excessive magenta edge contamination")

        if frame_failures:
            failures.extend(f"frame {index}: {failure}" for failure in frame_failures)
        frames.append(
            {
                "frame": index,
                "ok": not frame_failures,
                "bounds": [left, top, right, bottom],
                "area": int(visible.sum()),
                "components": sorted(components, reverse=True)[:4],
                "enclosed_transparent": enclosed_transparent(visible[top:bottom, left:right]),
                "failures": frame_failures,
            }
        )

    for anchor, index, label in ((args.first_anchor, 0, "first"), (args.last_anchor, frame_count - 1, "last")):
        if anchor is None:
            continue
        expected = Image.open(anchor).convert("RGBA")
        if expected.size != (CELL_W, CELL_H):
            failures.append(f"{label}_anchor size is {expected.size}, expected {(CELL_W, CELL_H)}")
            continue
        row, column = divmod(index, args.columns)
        actual = image.crop((column * CELL_W, row * CELL_H, (column + 1) * CELL_W, (row + 1) * CELL_H)).convert("RGBA")
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            failures.append(f"{label}_anchor does not exactly match {anchor}")

    report = {"ok": not failures, "atlas": str(args.atlas), "frame_count": frame_count, "columns": args.columns, "rows": args.rows, "frames": frames, "failures": failures}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
