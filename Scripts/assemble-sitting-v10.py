#!/usr/bin/env python3
"""Assemble 16 hard gaze anchors plus three generated mids per edge into 64 cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CELL_WIDTH = 192
CELL_HEIGHT = 208
COLUMNS = 8
FRAME_COUNT = 64
BODY_START_Y = 82


def load(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.size != (CELL_WIDTH, CELL_HEIGHT):
        raise ValueError(f"{path}: expected 192x208, got {image.size}")
    return image


def alpha_mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)[..., 3] > 8


def bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError("blank frame")
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 1.0


def body_pink_fraction(image: Image.Image, body_binary: np.ndarray) -> float:
    rgb = np.asarray(image, dtype=np.uint8)[..., :3][body_binary].astype(np.int16)
    pink = (rgb[:, 0] - rgb[:, 1] > 24) & (rgb[:, 0] - rgb[:, 2] > 24)
    return float(pink.mean()) if len(rgb) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, default=Path("Art/QA/sitting-v10"))
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Optional pre-regularized directory containing 00.png through 63.png.",
    )
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--contact", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    frames: list[Image.Image] = []
    sources: list[str] = []
    if args.frames_dir:
        for index in range(FRAME_COUNT):
            path = args.frames_dir / f"{index:02d}.png"
            frames.append(load(path))
            sources.append(str(path))
    else:
        for anchor_index in range(16):
            next_index = (anchor_index + 1) % 16
            anchor_path = args.qa_root / "anchors" / f"{anchor_index:02d}.png"
            frames.append(load(anchor_path))
            sources.append(str(anchor_path))
            pair_dir = args.qa_root / f"pair-{anchor_index:02d}-{next_index:02d}-registered"
            for generated_index in range(1, 4):
                path = pair_dir / f"{generated_index:02d}.png"
                frames.append(load(path))
                sources.append(str(path))

    if len(frames) != FRAME_COUNT:
        raise AssertionError(f"assembled {len(frames)} frames")

    rows = FRAME_COUNT // COLUMNS
    atlas = Image.new("RGBA", (CELL_WIDTH * COLUMNS, CELL_HEIGHT * rows), (0, 0, 0, 0))
    for index, frame in enumerate(frames):
        atlas.alpha_composite(frame, ((index % COLUMNS) * CELL_WIDTH, (index // COLUMNS) * CELL_HEIGHT))

    masks = [alpha_mask(frame) for frame in frames]
    body_masks = [item.copy() for item in masks]
    for item in body_masks:
        item[:BODY_START_Y, :] = False
    adjacent_body_iou = [iou(body_masks[index], body_masks[(index + 1) % FRAME_COUNT]) for index in range(FRAME_COUNT)]
    boxes = [bbox(item) for item in masks]
    widths = [right - left for left, _, right, _ in boxes]
    heights = [bottom - top for _, top, _, bottom in boxes]
    areas = [int(item.sum()) for item in masks]
    body_pink = [body_pink_fraction(frame, body) for frame, body in zip(frames, body_masks)]

    report = {
        "schema": "desktop-pet.sitting-v10.64-direction.v1",
        "frame_count": FRAME_COUNT,
        "columns": COLUMNS,
        "rows": rows,
        "body_start_y": BODY_START_Y,
        "minimum_adjacent_body_iou": min(adjacent_body_iou),
        "closing_body_iou_63_to_0": adjacent_body_iou[-1],
        "width_range": [min(widths), max(widths)],
        "height_range": [min(heights), max(heights)],
        "alpha_area_range": [min(areas), max(areas)],
        "maximum_body_pink_fraction": max(body_pink),
        "passes_body_iou": min(adjacent_body_iou) >= 0.95,
        "passes_body_pink": max(body_pink) <= 0.08,
        "adjacent_body_iou": adjacent_body_iou,
        "bounding_boxes": boxes,
        "sources": sources,
    }

    thumb_scale = 1
    label_height = 20
    contact = Image.new("RGBA", (CELL_WIDTH * COLUMNS, (CELL_HEIGHT + label_height) * rows), (52, 52, 55, 255))
    draw = ImageDraw.Draw(contact)
    for index, frame in enumerate(frames):
        x = (index % COLUMNS) * CELL_WIDTH
        y = (index // COLUMNS) * (CELL_HEIGHT + label_height)
        contact.alpha_composite(frame, (x, y + label_height))
        draw.text((x + 4, y + 3), f"{index:02d} {widths[index]}x{heights[index]}", fill=(255, 255, 255, 255))

    preview_frames: list[Image.Image] = []
    preview_background = (52, 52, 55, 255)
    for index, frame in enumerate(frames):
        preview = Image.new("RGBA", (CELL_WIDTH * 2, CELL_HEIGHT * 2), preview_background)
        preview.alpha_composite(frame.resize((CELL_WIDTH * 2, CELL_HEIGHT * 2), Image.Resampling.LANCZOS))
        ImageDraw.Draw(preview).text((8, 8), f"direction {index:02d}/63", fill=(255, 255, 255, 255))
        preview_frames.append(preview.convert("RGB"))

    for path in (args.atlas, args.contact, args.preview, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(args.atlas)
    contact.save(args.contact)
    preview_frames[0].save(
        args.preview,
        save_all=True,
        append_images=preview_frames[1:],
        duration=45,
        loop=0,
        optimize=False,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "frame_count", "minimum_adjacent_body_iou", "closing_body_iou_63_to_0",
        "width_range", "height_range", "alpha_area_range", "maximum_body_pink_fraction",
        "passes_body_iou", "passes_body_pink"
    )}, indent=2))


if __name__ == "__main__":
    main()
