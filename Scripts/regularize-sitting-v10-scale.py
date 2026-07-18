#!/usr/bin/env python3
"""Smooth sitting-v10 boundary scale pulses with whole-cat transforms only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


CELL_SIZE = (192, 208)
BASELINE_Y = 204
ALPHA_THRESHOLD = 8


def load(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.size != CELL_SIZE:
        raise ValueError(f"{path}: expected {CELL_SIZE}, got {image.size}")
    return image


def metrics(image: Image.Image) -> dict[str, float | int | list[int]]:
    alpha = np.asarray(image, dtype=np.uint8)[..., 3]
    visible = alpha > ALPHA_THRESHOLD
    ys, xs = np.where(visible)
    if not len(xs):
        raise ValueError("blank frame")
    weights = alpha[visible].astype(np.float64)
    return {
        "area": int(visible.sum()),
        "alpha_centroid_x": float(np.average(xs, weights=weights)),
        "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
    }


def transform_whole_cat(image: Image.Image, scale: float, target_center_x: float) -> Image.Image:
    alpha = np.asarray(image, dtype=np.uint8)[..., 3]
    ys, xs = np.where(alpha > ALPHA_THRESHOLD)
    crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    resized = crop.resize(size, Image.Resampling.LANCZOS)
    resized_alpha = np.asarray(resized, dtype=np.uint8)[..., 3]
    rys, rxs = np.where(resized_alpha > ALPHA_THRESHOLD)
    local_center_x = (int(rxs.min()) + int(rxs.max()) + 1) / 2
    x = round(target_center_x - local_center_x)
    y = BASELINE_Y - (int(rys.max()) + 1)
    cell = Image.new("RGBA", CELL_SIZE, (0, 0, 0, 0))
    cell.alpha_composite(resized, (x, y))
    arr = np.asarray(cell, dtype=np.uint8).copy()
    arr[arr[..., 3] == 0, :3] = 0
    return Image.fromarray(arr, "RGBA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, default=Path("Art/QA/sitting-v10"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-scale-change", type=float, default=0.04)
    parser.add_argument("--max-output-width", type=int, default=138)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    output_index = 0
    for anchor_index in range(16):
        next_index = (anchor_index + 1) % 16
        start_path = args.qa_root / "anchors" / f"{anchor_index:02d}.png"
        end_path = args.qa_root / "anchors" / f"{next_index:02d}.png"
        start = load(start_path)
        end = load(end_path)
        start_metrics = metrics(start)
        end_metrics = metrics(end)
        start.save(args.output_dir / f"{output_index:02d}.png")
        records.append({"frame": output_index, "kind": "hard_anchor", "source": str(start_path), "scale": 1.0})
        output_index += 1

        pair_dir = args.qa_root / f"pair-{anchor_index:02d}-{next_index:02d}-registered"
        for mid_index, t in enumerate((0.25, 0.5, 0.75), start=1):
            source_path = pair_dir / f"{mid_index:02d}.png"
            source = load(source_path)
            source_metrics = metrics(source)
            log_target_area = (1 - t) * math.log(int(start_metrics["area"])) + t * math.log(int(end_metrics["area"]))
            target_area = math.exp(log_target_area)
            scale = math.sqrt(target_area / int(source_metrics["area"]))
            source_box = source_metrics["box"]
            source_width = int(source_box[2]) - int(source_box[0])
            scale = min(scale, args.max_output_width / source_width)
            if abs(scale - 1) > args.max_scale_change:
                raise ValueError(
                    f"{source_path}: required whole-cat scale {scale:.5f} exceeds "
                    f"limit ±{args.max_scale_change:.3f}"
                )
            start_box = start_metrics["box"]
            end_box = end_metrics["box"]
            start_center_x = (int(start_box[0]) + int(start_box[2])) / 2
            end_center_x = (int(end_box[0]) + int(end_box[2])) / 2
            target_center_x = (1 - t) * start_center_x + t * end_center_x
            output = transform_whole_cat(source, scale, target_center_x)
            output_metrics = metrics(output)
            output.save(args.output_dir / f"{output_index:02d}.png")
            records.append(
                {
                    "frame": output_index,
                    "kind": "generated_mid_whole_cat_regularized",
                    "source": str(source_path),
                    "t": t,
                    "scale": scale,
                    "source_area": source_metrics["area"],
                    "target_area": target_area,
                    "output_area": output_metrics["area"],
                    "target_center_x": target_center_x,
                    "output_metrics": output_metrics,
                }
            )
            output_index += 1

    if output_index != 64:
        raise AssertionError(f"expected 64 frames, wrote {output_index}")
    scales = [float(item["scale"]) for item in records if item["kind"] != "hard_anchor"]
    report = {
        "schema": "desktop-pet.sitting-v10.whole-cat-scale-regularization.v1",
        "frame_count": output_index,
        "method": "complete-cat uniform scale and translation only; hard anchors unchanged",
        "scale_range": [min(scales), max(scales)],
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frame_count": output_index, "scale_range": report["scale_range"]}, indent=2))


if __name__ == "__main__":
    main()
