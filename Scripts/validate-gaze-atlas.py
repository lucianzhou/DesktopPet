#!/usr/bin/env python3
"""Validate a final 8-column gaze atlas before it can enter the app bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load_module("prepare_canonical", "prepare-canonical.py")
QA = load_module("validate_gaze_source", "validate-gaze-source.py")
FIXED_TORSO_START_Y = 110


def fixed_torso_mask(cell: Image.Image) -> np.ndarray:
    """Mask the part that must not move when only head/eyes track a pointer."""

    mask = np.asarray(cell.convert("RGBA"))[..., 3] > QA.VISIBLE_ALPHA
    mask[:FIXED_TORSO_START_Y] = False
    return mask


def mask_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument(
        "--allow-pose-silhouette",
        action="store_true",
        help=(
            "Allow a deliberate head-pose silhouette while strictly enforcing one fixed "
            "body width, baseline, and body-area range across all directions."
        ),
    )
    args = parser.parse_args()

    atlas = Image.open(args.atlas).convert("RGBA")
    canonical = Image.open(args.canonical).convert("RGBA")
    if canonical.size != PREP.CELL_SIZE:
        canonical, _ = PREP.register_cell(PREP.remove_magenta_key(canonical))
    canonical_ref = QA.canonical_body(canonical)
    canonical_visible = np.asarray(canonical)[..., 3] > QA.VISIBLE_ALPHA
    failures: list[str] = []
    cells: list[dict[str, object]] = []
    torso_masks: list[np.ndarray] = []
    torso_entries: list[dict[str, object]] = []
    expected_width = PREP.CELL_SIZE[0] * 8
    if atlas.width != expected_width or atlas.height <= 0 or atlas.height % PREP.CELL_SIZE[1] != 0:
        failures.append(
            f"atlas_size_{atlas.size}_expected_width_{expected_width}_and_whole_{PREP.CELL_SIZE[1]}px_rows"
        )
    row_count = atlas.height // PREP.CELL_SIZE[1] if atlas.height > 0 else 0

    for row in range(row_count):
        for column in range(8):
            cell = atlas.crop((column * PREP.CELL_SIZE[0], row * PREP.CELL_SIZE[1], (column + 1) * PREP.CELL_SIZE[0], (row + 1) * PREP.CELL_SIZE[1]))
            alpha = np.asarray(cell)[..., 3]
            visible = alpha > QA.VISIBLE_ALPHA
            cell_failures: list[str] = []
            if not visible.any():
                cell_failures.append("blank")
                cells.append({"row": row, "column": column, "failures": cell_failures})
                failures.append(f"cell_{row}_{column}:blank")
                continue
            components = QA.connected_components(visible)
            if len(components) != 1:
                cell_failures.append(f"component_count_{len(components)}")
            ys, xs = np.where(visible)
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            top_margin = 4 if args.allow_pose_silhouette else 8
            if box[0] < 8 or box[2] > 184 or box[1] < top_margin or box[3] != PREP.BASELINE_Y:
                cell_failures.append(f"registered_bounds_{box}")
            if max(alpha[:, 0]) or max(alpha[:, -1]) or max(alpha[0, :]) or max(alpha[-1, :]):
                cell_failures.append("touches_cell_edge")
            if not args.allow_pose_silhouette and not np.array_equal(visible, canonical_visible):
                cell_failures.append("canonical_silhouette_changed")
            metrics = QA.body_metrics(cell, canonical_ref)
            # A motion atlas may intentionally lower/raise the head, but its
            # shoulders, paws and tail must remain recognisably the canonical
            # cat.  The pose mode is deliberately far tighter than the former
            # 0.82...1.18 generated-strip gate.
            min_iou = 0.90 if args.allow_pose_silhouette else 0.82
            min_area = 0.88 if args.allow_pose_silhouette else 0.82
            max_area = 1.05 if args.allow_pose_silhouette else 1.18
            if metrics["body_iou"] < min_iou or not min_area <= metrics["body_area_ratio"] <= max_area:
                cell_failures.append("identity_or_body_registration_drift")
            pink = QA.pink_edge_pixels(cell)
            if pink:
                cell_failures.append(f"key_edge_spill_{pink}")
            if args.allow_pose_silhouette:
                torso = fixed_torso_mask(cell)
                torso_box = mask_box(torso)
                torso_masks.append(torso)
                torso_entries.append(
                    {
                        "row": row,
                        "column": column,
                        "box": torso_box,
                        "area": int(torso.sum()),
                        "center_x": round((torso_box[0] + torso_box[2]) / 2, 3),
                        "width": torso_box[2] - torso_box[0],
                    }
                )
            if cell_failures:
                failures.extend(f"cell_{row}_{column}:{failure}" for failure in cell_failures)
            cells.append({"row": row, "column": column, "box": box, "metrics": metrics, "pink_edge_pixels": pink, "failures": cell_failures})

    pose_geometry: dict[str, object] | None = None
    if args.allow_pose_silhouette and cells:
        # The v3 regression that prompted this gate narrowed the complete cat
        # from 136px to 123px while walking around the circle.  A valid
        # pose-preserving atlas must instead keep the lower-body footprint
        # steady; changing eyes/head may alter only the upper silhouette.
        widths = [int(cell["box"][2]) - int(cell["box"][0]) for cell in cells if "box" in cell]
        areas = [float(cell["metrics"]["body_area_ratio"]) for cell in cells if "metrics" in cell]
        max_width_delta = max(widths) - min(widths) if widths else 0
        max_area_delta = max(areas) - min(areas) if areas else 0.0
        adjacent_area_delta = max(
            abs(areas[index] - areas[(index + 1) % len(areas)])
            for index in range(len(areas))
        ) if areas else 0.0
        high_density = len(cells) >= 32
        max_full_width_delta = 4 if high_density else 1
        max_body_area_delta = 0.06 if high_density else 0.05
        max_adjacent_body_area_delta = 0.02 if high_density else 0.03
        max_torso_left_delta = 2 if high_density else 1
        max_torso_width_delta = 4 if high_density else 1
        max_torso_area_spread = 0.05 if high_density else 0.035
        pose_geometry = {
            "full_sprite_widths": widths,
            "max_full_sprite_width_delta": max_width_delta,
            "body_area_ratios": [round(value, 5) for value in areas],
            "max_body_area_delta": round(max_area_delta, 5),
            "max_adjacent_body_area_delta_including_loop": round(adjacent_area_delta, 5),
            "limits": {
                "max_full_sprite_width_delta": max_full_width_delta,
                "max_body_area_delta": max_body_area_delta,
                "max_adjacent_body_area_delta": max_adjacent_body_area_delta,
            },
        }
        if max_width_delta > max_full_width_delta:
            failures.append(f"pose_geometry:full_sprite_width_drift_{max_width_delta}px")
        if max_area_delta > max_body_area_delta:
            failures.append(f"pose_geometry:body_area_drift_{max_area_delta:.5f}")
        if adjacent_area_delta > max_adjacent_body_area_delta:
            failures.append(f"pose_geometry:adjacent_body_area_drift_{adjacent_area_delta:.5f}")

        # Compare only the lower torso, forepaws and tail across neighbours.
        # Head pitch is allowed to change height, but this region must neither
        # slide, compress nor widen as the pointer crosses a direction seam.
        torso_areas = [int(item["area"]) for item in torso_entries]
        torso_lefts = [int(item["box"][0]) for item in torso_entries]
        torso_centers = [float(item["center_x"]) for item in torso_entries]
        torso_widths = [int(item["width"]) for item in torso_entries]
        torso_area_spread = (max(torso_areas) - min(torso_areas)) / max(torso_areas) if torso_areas else 0.0
        torso_adjacent_ious: list[float] = []
        for index, mask in enumerate(torso_masks):
            next_mask = torso_masks[(index + 1) % len(torso_masks)]
            intersection = int(np.logical_and(mask, next_mask).sum())
            union = int(np.logical_or(mask, next_mask).sum())
            torso_adjacent_ious.append(intersection / union if union else 0.0)
        torso_min_iou = min(torso_adjacent_ious) if torso_adjacent_ious else 0.0
        pose_geometry["fixed_torso"] = {
            "y_start": FIXED_TORSO_START_Y,
            "boxes": [item["box"] for item in torso_entries],
            "areas": torso_areas,
            "max_left_delta": max(torso_lefts) - min(torso_lefts) if torso_lefts else 0,
            "max_center_x_delta": round(max(torso_centers) - min(torso_centers), 5) if torso_centers else 0.0,
            "max_width_delta": max(torso_widths) - min(torso_widths) if torso_widths else 0,
            "relative_area_spread": round(torso_area_spread, 5),
            "adjacent_ious_including_loop": [round(value, 5) for value in torso_adjacent_ious],
            "min_adjacent_iou_including_loop": round(torso_min_iou, 5),
            "limits": {
                "max_left_delta": max_torso_left_delta,
                "max_center_x_delta": 1.0,
                "max_width_delta": max_torso_width_delta,
                "relative_area_spread": max_torso_area_spread,
                "min_adjacent_iou_including_loop": 0.95,
            },
        }
        if torso_lefts and max(torso_lefts) - min(torso_lefts) > max_torso_left_delta:
            failures.append("pose_geometry:fixed_torso_left_drift")
        if torso_centers and max(torso_centers) - min(torso_centers) > 1.0:
            failures.append("pose_geometry:fixed_torso_center_drift")
        if torso_widths and max(torso_widths) - min(torso_widths) > max_torso_width_delta:
            failures.append("pose_geometry:fixed_torso_width_drift")
        if torso_area_spread > max_torso_area_spread:
            failures.append(f"pose_geometry:fixed_torso_area_drift_{torso_area_spread:.5f}")
        if torso_min_iou < 0.95:
            failures.append(f"pose_geometry:fixed_torso_adjacent_iou_{torso_min_iou:.5f}")

    report = {
        "ok": not failures,
        "atlas": str(args.atlas),
        "canonical": str(args.canonical),
        "mode": "pose-preserving" if args.allow_pose_silhouette else "exact-canonical-silhouette",
        "cells": cells,
        "pose_geometry": pose_geometry,
        "failures": failures,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failure_count": len(failures)}))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
