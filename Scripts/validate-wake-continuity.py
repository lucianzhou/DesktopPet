#!/usr/bin/env python3
"""Measure identity, scale, registration, and row-boundary continuity for wake atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CELL_W, CELL_H = 192, 208
VISIBLE_ALPHA = 8


def metrics(cell: Image.Image) -> dict[str, object]:
    rgba = np.asarray(cell.convert("RGBA"), dtype=np.uint8)
    visible = rgba[..., 3] > VISIBLE_ALPHA
    ys, xs = np.where(visible)
    if len(xs) == 0:
        raise ValueError("blank frame")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    height = bottom - top
    head_limit = top + max(1, round(height * 0.42))
    widths = visible[top:head_limit].sum(axis=1)
    widths = widths[widths > 0]
    head_width = float(np.percentile(widths, 85)) if len(widths) else 0.0
    opaque = rgba[..., 3] > 128
    colors = rgba[..., :3][opaque]
    median_rgb = [float(value) for value in np.median(colors, axis=0)] if len(colors) else [0.0, 0.0, 0.0]
    return {
        "box": [left, top, right, bottom],
        "width": right - left,
        "height": height,
        "area": int(visible.sum()),
        "center_x": (left + right) / 2.0,
        "head_width": round(head_width, 4),
        "median_rgb": [round(value, 3) for value in median_rgb],
    }


def ratio(a: float, b: float) -> float:
    return max(a, b) / max(1e-9, min(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--first-anchor", type=Path)
    parser.add_argument("--last-anchor", type=Path)
    parser.add_argument(
        "--continuous-rows",
        action="store_true",
        help="Treat row boundaries as ordinary consecutive frames instead of duplicated strip endpoints.",
    )
    args = parser.parse_args()

    atlas = Image.open(args.atlas).convert("RGBA")
    expected = (args.columns * CELL_W, args.rows * CELL_H)
    failures: list[str] = []
    warnings: list[str] = []
    if atlas.size != expected:
        failures.append(f"atlas_size={atlas.size}; expected={expected}")

    cells: list[Image.Image] = []
    frame_metrics: list[dict[str, object]] = []
    for index in range(args.columns * args.rows):
        row, column = divmod(index, args.columns)
        cell = atlas.crop((column * CELL_W, row * CELL_H, (column + 1) * CELL_W, (row + 1) * CELL_H))
        cells.append(cell)
        frame_metrics.append(metrics(cell))

    transitions: list[dict[str, object]] = []
    for index in range(len(frame_metrics) - 1):
        before = frame_metrics[index]
        after = frame_metrics[index + 1]
        head_ratio = ratio(float(before["head_width"]), float(after["head_width"]))
        area_ratio = ratio(float(before["area"]), float(after["area"]))
        width_ratio = ratio(float(before["width"]), float(after["width"]))
        height_ratio = ratio(float(before["height"]), float(after["height"]))
        center_shift = abs(float(before["center_x"]) - float(after["center_x"]))
        color_delta = max(
            abs(float(a) - float(b))
            for a, b in zip(before["median_rgb"], after["median_rgb"])
        )
        transition_failures: list[str] = []
        transition_warnings: list[str] = []
        if head_ratio > 1.20:
            transition_warnings.append(f"head_width_ratio={head_ratio:.3f}")
        if area_ratio > 1.32:
            transition_warnings.append(f"visible_area_ratio={area_ratio:.3f}")
        if width_ratio > 1.24:
            transition_warnings.append(f"width_ratio={width_ratio:.3f}")
        if height_ratio > 1.38:
            transition_warnings.append(f"height_ratio={height_ratio:.3f}")
        if center_shift > 12:
            transition_failures.append(f"center_x_shift={center_shift:.3f}")
        if color_delta > 18:
            transition_warnings.append(f"median_rgb_delta={color_delta:.3f}")
        if transition_failures:
            failures.extend(f"F{index:02d}->F{index + 1:02d}:{failure}" for failure in transition_failures)
        if transition_warnings:
            warnings.extend(f"F{index:02d}->F{index + 1:02d}:{warning}" for warning in transition_warnings)
        transitions.append(
            {
                "from": index,
                "to": index + 1,
                "head_width_ratio": round(head_ratio, 5),
                "area_ratio": round(area_ratio, 5),
                "width_ratio": round(width_ratio, 5),
                "height_ratio": round(height_ratio, 5),
                "center_x_shift": round(center_shift, 5),
                "median_rgb_delta": round(color_delta, 5),
                "failures": transition_failures,
                "warnings": transition_warnings,
            }
        )

    duplicate_boundaries = []
    if not args.continuous_rows:
        for row in range(1, args.rows):
            left = row * args.columns - 1
            right = row * args.columns
            equal = np.array_equal(np.asarray(cells[left]), np.asarray(cells[right]))
            duplicate_boundaries.append({"left": left, "right": right, "exact": equal})
            if not equal:
                failures.append(f"shared_row_boundary_F{left:02d}_F{right:02d}_not_exact")

    for anchor_path, index, label in (
        (args.first_anchor, 0, "first"),
        (args.last_anchor, len(cells) - 1, "last"),
    ):
        if anchor_path is None:
            continue
        anchor = Image.open(anchor_path).convert("RGBA")
        if anchor.size != (CELL_W, CELL_H) or not np.array_equal(np.asarray(anchor), np.asarray(cells[index])):
            failures.append(f"{label}_anchor_mismatch")

    report = {
        "schema": "desktop-pet.baomihua-wake-continuity.v1",
        "ok": not failures,
        "atlas": str(args.atlas),
        "frames": frame_metrics,
        "transitions": transitions,
        "duplicate_boundaries": duplicate_boundaries,
        "warnings": warnings,
        "failures": failures,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": not failures, "failure_count": len(failures), "warning_count": len(warnings), "json_out": str(args.json_out)}))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
