#!/usr/bin/env python3
"""Validate one five-frame generated sitting-gaze bridge against hard anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CELL_SIZE = (192, 208)
BODY_START_Y = 82
ALPHA_THRESHOLD = 8


def load(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.size != CELL_SIZE:
        raise ValueError(f"{path} has size {image.size}, expected {CELL_SIZE}")
    return image


def mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)[..., 3] > ALPHA_THRESHOLD


def bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 1.0


def body_color_metrics(image: Image.Image, body_binary: np.ndarray) -> tuple[list[int], float]:
    rgb = np.asarray(image, dtype=np.uint8)[..., :3]
    pixels = rgb[body_binary]
    median = np.median(pixels, axis=0).round().astype(int).tolist()
    signed = pixels.astype(np.int16)
    pink = (signed[:, 0] - signed[:, 1] > 24) & (signed[:, 0] - signed[:, 2] > 24)
    return median, float(pink.mean()) if len(pixels) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-anchor", required=True, type=Path)
    parser.add_argument("--end-anchor", required=True, type=Path)
    parser.add_argument("--registered-dir", required=True, type=Path)
    parser.add_argument("--contact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    generated = [load(args.registered_dir / f"{index:02d}.png") for index in range(5)]
    frames = [load(args.start_anchor), generated[1], generated[2], generated[3], load(args.end_anchor)]
    masks = [mask(frame) for frame in frames]
    body_masks = [item.copy() for item in masks]
    for item in body_masks:
        item[:BODY_START_Y, :] = False

    adjacent_body_iou = [iou(body_masks[index], body_masks[index + 1]) for index in range(4)]
    boxes = [bbox(item) for item in masks]
    widths = [right - left for left, _, right, _ in boxes]
    heights = [bottom - top for _, top, _, bottom in boxes]
    areas = [int(item.sum()) for item in masks]
    color_metrics = [body_color_metrics(frame, body) for frame, body in zip(frames, body_masks)]
    body_median_rgb = [item[0] for item in color_metrics]
    body_pink_fraction = [item[1] for item in color_metrics]
    report = {
        "frames": [str(args.start_anchor), *[str(args.registered_dir / f"{i:02d}.png") for i in range(1, 4)], str(args.end_anchor)],
        "body_start_y": BODY_START_Y,
        "adjacent_body_iou": adjacent_body_iou,
        "minimum_adjacent_body_iou": min(adjacent_body_iou),
        "bounding_boxes": boxes,
        "widths": widths,
        "heights": heights,
        "alpha_areas": areas,
        "body_median_rgb": body_median_rgb,
        "body_pink_fraction": body_pink_fraction,
        "maximum_body_pink_fraction": max(body_pink_fraction),
        "passes_body_iou": min(adjacent_body_iou) >= 0.95,
        "passes_body_pink": max(body_pink_fraction) <= 0.08,
    }

    scale = 2
    label_height = 30
    contact = Image.new("RGBA", (CELL_SIZE[0] * scale * 5, (CELL_SIZE[1] + label_height) * scale), (52, 52, 55, 255))
    draw = ImageDraw.Draw(contact)
    for index, frame in enumerate(frames):
        enlarged = frame.resize((CELL_SIZE[0] * scale, CELL_SIZE[1] * scale), Image.Resampling.NEAREST)
        x = index * CELL_SIZE[0] * scale
        contact.alpha_composite(enlarged, (x, label_height * scale))
        label = f"{index}  {widths[index]}x{heights[index]}  area={areas[index]}"
        draw.text((x + 8, 8), label, fill=(255, 255, 255, 255))
        if index:
            draw.text((x + 8, 30), f"body IoU={adjacent_body_iou[index - 1]:.4f}", fill=(170, 255, 170, 255))

    args.contact.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    contact.save(args.contact)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
