#!/usr/bin/env python3
"""Strict QA gate for a prepared Baomihua v5 interaction atlas.

The validator does more than inspect a finished PNG.  It rereads the single
source storyboard, enforces the explicitly declared 8x8/64 or 8x6/48 layout,
rebuilds the expected atlas with the recorded one-scale transform, and compares
all frame pixels in row-major order.  That makes accidental strip reordering,
per-frame fitting, duplicated neutral frames, and hidden face/body patch paths
observable rather than a matter of trust.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_prepare_module() -> Any:
    path = Path(__file__).with_name("prepare-interaction-v5.py")
    spec = importlib.util.spec_from_file_location("prepare_interaction_v5", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their annotations through sys.modules while the
    # dynamically loaded preparation module is executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREP = load_prepare_module()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PREP.jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_prepare_report(path: Path, failures: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        failures.append(f"prepare_report_unreadable:{error}")
        return None
    except json.JSONDecodeError as error:
        failures.append(f"prepare_report_invalid_json:{error}")
        return None
    if not isinstance(payload, dict):
        failures.append("prepare_report_is_not_a_json_object")
        return None
    if payload.get("schema") != "desktop-pet.interaction-v5.prepare.v1":
        failures.append(f"prepare_report_schema={payload.get('schema')!r}; expected desktop-pet.interaction-v5.prepare.v1")
    if payload.get("ok") is not True:
        failures.append("prepare_report_did_not_record_a_successful_preparation")
    return payload


def crop_cell(atlas: Image.Image, index: int, *, columns: int) -> Image.Image:
    row, column = divmod(index, columns)
    return atlas.crop(
        (
            column * PREP.CELL_WIDTH,
            row * PREP.CELL_HEIGHT,
            (column + 1) * PREP.CELL_WIDTH,
            (row + 1) * PREP.CELL_HEIGHT,
        )
    )


def validate_byte_identical_frame_repetitions(
    atlas: Image.Image,
    *,
    frame_count: int,
    columns: int,
    failures: list[str],
) -> list[dict[str, Any]]:
    """Allow only the exact first/final neutral anchor pair to repeat.

    Replaying a single action cell to simulate a higher frame rate is not a
    real interaction pose.  The comparison intentionally uses full RGBA bytes,
    so equal-looking but differently encoded PNG chunks cannot evade the gate.
    """

    buckets: dict[bytes, list[int]] = {}
    for index in range(frame_count):
        rgba_bytes = crop_cell(atlas, index, columns=columns).convert("RGBA").tobytes()
        buckets.setdefault(rgba_bytes, []).append(index)
    groups: list[dict[str, Any]] = []
    final_index = frame_count - 1
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        frame_ids = [f"F{index:02d}" for index in indices]
        allowed_anchor_pair = indices == [0, final_index]
        group = {
            "frames": frame_ids,
            "allowed": allowed_anchor_pair,
            "reason": "first_final_neutral_anchor_pair" if allowed_anchor_pair else "duplicate_action_or_non_anchor_frame",
        }
        groups.append(group)
        if not allowed_anchor_pair:
            if any(0 < index < final_index for index in indices):
                failures.append("byte_identical_duplicate_interior_action_frames:" + ",".join(frame_ids))
            else:
                failures.append("byte_identical_duplicate_frames_not_allowed:" + ",".join(frame_ids))
    return groups


def is_valid_padding(metrics: dict[str, Any]) -> bool:
    if metrics.get("blank"):
        return False
    padding = metrics["padding"]
    return (
        padding["left"] >= PREP.LEFT_PADDING
        and padding["right"] >= PREP.RIGHT_PADDING
        and padding["top"] >= PREP.TOP_PADDING
        and padding["bottom"] >= PREP.BOTTOM_PADDING
    )


def validate_atlas_cells(
    atlas: Image.Image,
    *,
    rows: int,
    columns: int,
    failures: list[str],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Check every output cell before any temporal continuity interpretation."""

    metrics_records: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    required_padding = {
        "left": PREP.LEFT_PADDING,
        "right": PREP.RIGHT_PADDING,
        "top": PREP.TOP_PADDING,
        "bottom": PREP.BOTTOM_PADDING,
    }
    for index in range(rows * columns):
        cell = crop_cell(atlas, index, columns=columns)
        metrics = PREP.cell_metrics(cell)
        metrics_records.append({"index": index, "frame": f"F{index:02d}", "metrics": metrics})
        array = np.asarray(cell.convert("RGBA"), dtype=np.uint8)
        masks.append(array[..., 3] > PREP.VISIBLE_ALPHA)
        frame_failures: list[str] = []
        if metrics["blank"]:
            frame_failures.append("blank")
        else:
            if metrics["component_count"] != 1:
                frame_failures.append(f"component_count={metrics['component_count']}; expected 1")
            if metrics["baseline_y"] != PREP.BASELINE_Y:
                frame_failures.append(f"baseline={metrics['baseline_y']}; expected {PREP.BASELINE_Y}")
            if not is_valid_padding(metrics):
                frame_failures.append(f"padding={metrics['padding']}; required={required_padding}")
        if metrics["transparent_rgb_pixels"]:
            frame_failures.append(f"transparent_rgb_not_cleared={metrics['transparent_rgb_pixels']}")
        if metrics["key_spill_pixels"]:
            frame_failures.append(f"magenta_key_spill={metrics['key_spill_pixels']}")
        failures.extend(f"frame_F{index:02d}:{reason}" for reason in frame_failures)
    return metrics_records, masks


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate without NumPy wraparound for centroid-aligned IoU."""

    output = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    if source_y0 >= source_y1 or source_x0 >= source_x1:
        return output
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    output[target_y0:target_y1, target_x0:target_x1] = mask[source_y0:source_y1, source_x0:source_x1]
    return output


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum()) / union


def _ratio(first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return float("inf")
    return max(first, second) / min(first, second)


def continuity_metrics(
    frames: list[dict[str, Any]],
    masks: list[np.ndarray],
) -> dict[str, Any]:
    """Reject pops while letting a smooth crouch→stand→crouch arc through.

    The checks deliberately use adjacent and local-neighbour ratios rather than
    forcing a constant height or area.  A real rise/stand/sit sequence can
    change volume gradually; a one-frame scale pop, identity swap, or centroid
    teleport cannot.
    """

    valid = all(not record["metrics"].get("blank") for record in frames)
    if not valid:
        return {"ok": False, "failures": ["continuity_unavailable_due_to_blank_frame"], "pairs": [], "local_outliers": []}

    pair_thresholds = {
        "area_ratio_max": 1.27,
        "width_ratio_max": 1.25,
        "height_ratio_max": 1.23,
        "centroid_dx_max": 24.0,
        "centroid_dy_max": 20.0,
        "aligned_iou_review_threshold": 0.23,
    }
    failures: list[str] = []
    pairs: list[dict[str, Any]] = []
    for index in range(len(frames) - 1):
        first = frames[index]["metrics"]
        second = frames[index + 1]["metrics"]
        area_ratio = _ratio(float(first["area"]), float(second["area"]))
        width_ratio = _ratio(float(first["width"]), float(second["width"]))
        height_ratio = _ratio(float(first["height"]), float(second["height"]))
        scale_proxy_ratio = math.sqrt(area_ratio)
        dx = abs(float(first["centroid"][0]) - float(second["centroid"][0]))
        dy = abs(float(first["centroid"][1]) - float(second["centroid"][1]))
        raw_iou = iou(masks[index], masks[index + 1])
        shift_x = int(round(float(first["centroid"][0]) - float(second["centroid"][0])))
        shift_y = int(round(float(first["centroid"][1]) - float(second["centroid"][1])))
        aligned_iou = iou(masks[index], shift_mask(masks[index + 1], shift_x, shift_y))
        pair_failures: list[str] = []
        pair_warnings: list[str] = []
        if area_ratio > pair_thresholds["area_ratio_max"]:
            pair_failures.append(f"area_ratio={area_ratio:.3f}")
        if width_ratio > pair_thresholds["width_ratio_max"]:
            pair_failures.append(f"width_ratio={width_ratio:.3f}")
        if height_ratio > pair_thresholds["height_ratio_max"]:
            pair_failures.append(f"height_ratio={height_ratio:.3f}")
        if dx > pair_thresholds["centroid_dx_max"]:
            pair_failures.append(f"centroid_dx={dx:.3f}")
        if dy > pair_thresholds["centroid_dy_max"]:
            pair_failures.append(f"centroid_dy={dy:.3f}")
        geometry_pop_signal = (
            area_ratio > 1.16
            or width_ratio > 1.14
            or height_ratio > 1.13
            or dx > 14.0
            or dy > 13.0
        )
        if aligned_iou < pair_thresholds["aligned_iou_review_threshold"]:
            if geometry_pop_signal:
                pair_failures.append(f"aligned_iou={aligned_iou:.3f}_with_geometry_pop")
            else:
                pair_warnings.append(f"aligned_iou={aligned_iou:.3f}_low_but_geometry_changes_are_gradual")
        record = {
            "from": f"F{index:02d}",
            "to": f"F{index + 1:02d}",
            "area_ratio": round(area_ratio, 6),
            "scale_proxy_ratio": round(scale_proxy_ratio, 6),
            "width_ratio": round(width_ratio, 6),
            "height_ratio": round(height_ratio, 6),
            "centroid_delta": [round(dx, 6), round(dy, 6)],
            "raw_iou": round(raw_iou, 6),
            "centroid_aligned_iou": round(aligned_iou, 6),
            "failures": pair_failures,
            "warnings": pair_warnings,
        }
        pairs.append(record)
        failures.extend(f"continuity_{record['from']}_{record['to']}:{reason}" for reason in pair_failures)

    # A gradual height/area arc is accepted.  An isolated frame that is far
    # outside the average of its immediate neighbours is not: it is the usual
    # signature of a different cat scale, a bad crop, or a frame swap.
    local_outliers: list[dict[str, Any]] = []
    for index in range(1, len(frames) - 1):
        previous = frames[index - 1]["metrics"]
        current = frames[index]["metrics"]
        following = frames[index + 1]["metrics"]
        values = {
            "area": (float(previous["area"]), float(current["area"]), float(following["area"]), 1.35),
            "width": (float(previous["width"]), float(current["width"]), float(following["width"]), 1.28),
            "height": (float(previous["height"]), float(current["height"]), float(following["height"]), 1.28),
        }
        reasons: list[str] = []
        for name, (before, value, after, threshold) in values.items():
            local_trend = (before + after) / 2.0
            ratio = _ratio(value, local_trend)
            if ratio > threshold:
                reasons.append(f"{name}_vs_local_trend={ratio:.3f}")
        if reasons:
            local_outliers.append({"frame": f"F{index:02d}", "failures": reasons})
            failures.extend(f"continuity_F{index:02d}_isolated:{reason}" for reason in reasons)

    # Wide global ranges are intentionally generous: a full stand can be
    # taller than a crouch.  They only reject an obviously different-size cat.
    middle = [record["metrics"] for record in frames[1:-1]] or [record["metrics"] for record in frames]
    global_ranges = {
        "area_ratio": _ratio(max(float(item["area"]) for item in middle), min(float(item["area"]) for item in middle)),
        "width_ratio": _ratio(max(float(item["width"]) for item in middle), min(float(item["width"]) for item in middle)),
        "height_ratio": _ratio(max(float(item["height"]) for item in middle), min(float(item["height"]) for item in middle)),
    }
    if global_ranges["area_ratio"] > 2.15:
        failures.append(f"global_body_area_range={global_ranges['area_ratio']:.3f}; obvious inconsistent body size")
    if global_ranges["width_ratio"] > 1.75:
        failures.append(f"global_body_width_range={global_ranges['width_ratio']:.3f}; obvious inconsistent body size")
    if global_ranges["height_ratio"] > 1.85:
        failures.append(f"global_body_height_range={global_ranges['height_ratio']:.3f}; obvious inconsistent body size")

    return {
        "ok": not failures,
        "policy": "adjacent + local trend gates permit a smooth rise/stand/sit arc but reject pops",
        "thresholds": pair_thresholds,
        "pairs": pairs,
        "local_outliers": local_outliers,
        "global_ranges": {name: round(value, 6) for name, value in global_ranges.items()},
        "failures": failures,
    }


def validate_report_contract(
    report: dict[str, Any] | None,
    *,
    storyboard: Path,
    neutral: Path,
    atlas: Path,
    rows: int,
    columns: int,
    failures: list[str],
) -> dict[str, Any]:
    """Verify the report explicitly documents a no-patch row-major transform."""

    result: dict[str, Any] = {"ok": False}
    if report is None:
        return result
    frame_count = rows * columns
    expected_size = [columns * PREP.CELL_WIDTH, rows * PREP.CELL_HEIGHT]
    geometry = report.get("atlas_geometry")
    if not isinstance(geometry, dict):
        failures.append("prepare_report_missing_atlas_geometry")
    else:
        for key, expected in (("rows", rows), ("columns", columns), ("frame_count", frame_count), ("size", expected_size), ("mode", "RGBA")):
            if geometry.get(key) != expected:
                failures.append(f"prepare_report_atlas_geometry_{key}={geometry.get(key)!r}; expected {expected!r}")
    hashes = (
        ("storyboard_sha256", storyboard),
        ("runtime_neutral_sha256", neutral),
        ("output_sha256", atlas),
    )
    for key, path in hashes:
        expected_hash = report.get(key)
        try:
            actual_hash = sha256(path)
        except OSError as error:
            failures.append(f"cannot_hash_{key}_input:{error}")
            continue
        if expected_hash != actual_hash:
            failures.append(f"prepare_report_{key}_mismatch")

    transform = report.get("shared_transform")
    shared_scale: float | None = None
    if not isinstance(transform, dict):
        failures.append("prepare_report_missing_shared_transform")
    else:
        try:
            shared_scale = float(transform.get("scale"))
        except (TypeError, ValueError):
            failures.append("prepare_report_shared_scale_missing_or_non_numeric")
        else:
            if not math.isfinite(shared_scale) or shared_scale <= 0:
                failures.append("prepare_report_shared_scale_not_positive")
        for key, expected in (("per_frame_fit", False), ("face_overlay", False), ("body_lock", False), ("neutral_may_not_be_duplicated_elsewhere", True)):
            if transform.get(key) is not expected:
                failures.append(f"prepare_report_shared_transform_{key}={transform.get(key)!r}; expected {expected!r}")
        if transform.get("shared_baseline_y") != PREP.BASELINE_Y:
            failures.append(f"prepare_report_baseline={transform.get('shared_baseline_y')}; expected {PREP.BASELINE_Y}")
        if transform.get("neutral_anchor_indices") != [0, frame_count - 1]:
            failures.append(f"prepare_report_neutral_anchor_indices={transform.get('neutral_anchor_indices')!r}; expected [0,{frame_count - 1}]")

    order = report.get("frame_order")
    if not isinstance(order, list) or len(order) != frame_count:
        failures.append(f"prepare_report_frame_order_length={len(order) if isinstance(order, list) else 'missing'}; expected {frame_count}")
    else:
        for index, item in enumerate(order):
            expected = {
                "index": index,
                "frame": f"F{index:02d}",
                "source_frame_index": index,
                "storyboard_slot": [index // columns, index % columns],
                "atlas_slot": [index // columns, index % columns],
            }
            if not isinstance(item, dict):
                failures.append(f"prepare_report_frame_order_F{index:02d}_not_object")
                continue
            for key, value in expected.items():
                if item.get(key) != value:
                    failures.append(f"prepare_report_frame_order_F{index:02d}_{key}={item.get(key)!r}; expected {value!r}")

    records = report.get("frames")
    if not isinstance(records, list) or len(records) != frame_count:
        failures.append(f"prepare_report_frames_length={len(records) if isinstance(records, list) else 'missing'}; expected {frame_count}")
    else:
        for index, item in enumerate(records):
            transform_record = item.get("transform") if isinstance(item, dict) else None
            if not isinstance(transform_record, dict):
                failures.append(f"prepare_report_frame_F{index:02d}_missing_transform")
                continue
            expected_kind = "exact_runtime_neutral_anchor" if index in (0, frame_count - 1) else "complete_storyboard_cat_shared_scale"
            if transform_record.get("kind") != expected_kind:
                failures.append(f"prepare_report_frame_F{index:02d}_kind={transform_record.get('kind')!r}; expected {expected_kind}")
            if index not in (0, frame_count - 1):
                try:
                    frame_scale = float(transform_record.get("scale"))
                except (TypeError, ValueError):
                    failures.append(f"prepare_report_frame_F{index:02d}_missing_shared_scale")
                else:
                    if shared_scale is not None and not math.isclose(frame_scale, shared_scale, rel_tol=1e-12, abs_tol=1e-12):
                        failures.append(f"prepare_report_frame_F{index:02d}_uses_per_frame_scale={frame_scale}")
                for key in ("per_frame_fit", "face_overlay", "body_lock"):
                    if transform_record.get(key) is not False:
                        failures.append(f"prepare_report_frame_F{index:02d}_{key}_not_false")
            elif transform_record.get("scale") is not None:
                failures.append(f"prepare_report_anchor_F{index:02d}_must_not_be_rescaled")
    result.update({"ok": not failures, "shared_scale": shared_scale, "frame_count": frame_count})
    return result


def reconstruction_check(
    report: dict[str, Any] | None,
    analysis: Any | None,
    neutral: Image.Image | None,
    atlas: Image.Image | None,
    *,
    failures: list[str],
) -> dict[str, Any]:
    """Recreate all pixels to prove source-to-atlas ordering and transform."""

    result: dict[str, Any] = {"attempted": False, "ok": False, "mismatched_frames": []}
    if report is None or analysis is None or neutral is None or atlas is None:
        result["reason"] = "missing_valid_report_source_neutral_or_atlas"
        return result
    result["attempted"] = True
    try:
        expected_scale, _ = PREP.compute_shared_scale(analysis)
        recorded_scale = float(report["shared_transform"]["scale"])
        if not math.isclose(expected_scale, recorded_scale, rel_tol=1e-12, abs_tol=1e-12):
            failures.append(
                f"prepare_report_shared_scale={recorded_scale:.12f}; deterministic_source_scale={expected_scale:.12f}"
            )
        expected, expected_records = PREP.build_atlas(analysis, neutral, expected_scale)
    except Exception as error:  # A source/transform violation must never be hidden.
        failures.append(f"unable_to_reconstruct_source_order:{error}")
        result["reason"] = str(error)
        return result
    expected_array = np.asarray(expected.convert("RGBA"), dtype=np.uint8)
    actual_array = np.asarray(atlas.convert("RGBA"), dtype=np.uint8)
    mismatches: list[dict[str, Any]] = []
    for index in range(analysis.frame_count):
        expected_cell = np.asarray(crop_cell(expected, index, columns=analysis.columns), dtype=np.uint8)
        actual_cell = np.asarray(crop_cell(atlas, index, columns=analysis.columns), dtype=np.uint8)
        if not np.array_equal(expected_cell, actual_cell):
            mismatches.append(
                {
                    "frame": f"F{index:02d}",
                    "different_rgba_pixels": int(np.count_nonzero(np.any(expected_cell != actual_cell, axis=2))),
                }
            )
    if mismatches:
        failures.extend(
            f"source_atlas_frame_order_or_transform_mismatch:{item['frame']}:{item['different_rgba_pixels']}px"
            for item in mismatches
        )
    # Report source metadata must correspond to the same recovered source
    # frames, not merely claim row-major ordering.
    report_frames = report.get("frames") if isinstance(report, dict) else None
    metadata_mismatches: list[str] = []
    if isinstance(report_frames, list) and len(report_frames) == analysis.frame_count:
        for index, expected_record in enumerate(expected_records):
            actual_record = report_frames[index]
            for key in ("index", "frame", "source_frame_index", "storyboard_slot", "atlas_slot", "source_component_label", "source_box"):
                if not isinstance(actual_record, dict) or actual_record.get(key) != expected_record.get(key):
                    metadata_mismatches.append(f"F{index:02d}.{key}")
    if metadata_mismatches:
        failures.append("prepare_report_source_metadata_mismatch:" + ",".join(metadata_mismatches))
    result.update(
        {
            "ok": not mismatches and not metadata_mismatches and np.array_equal(expected_array, actual_array),
            "mismatched_frames": mismatches,
            "metadata_mismatches": metadata_mismatches,
            "deterministic_shared_scale": expected_scale,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one v5 interaction atlas against its one-board source and preparation report.")
    parser.add_argument("--storyboard", required=True, type=Path, help="The same unsliced flat-magenta source given to preparation.")
    parser.add_argument("--rows", type=int, default=PREP.DEFAULT_ROWS, help="Explicit source/atlas rows: default 8, or 6 for the 48-frame board.")
    parser.add_argument("--columns", type=int, default=PREP.DEFAULT_COLUMNS, help="Explicit source/atlas columns: currently 8.")
    parser.add_argument("--neutral", required=True, type=Path, help="Exact neutral required at first/final output frames.")
    parser.add_argument("--atlas", required=True, type=Path, help="Prepared RGBA interaction atlas.")
    parser.add_argument("--prepare-report", required=True, type=Path, help="JSON emitted by prepare-interaction-v5.py.")
    parser.add_argument("--json-out", required=True, type=Path, help="Validator report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    try:
        PREP.require_supported_grid(args.rows, args.columns)
    except ValueError as error:
        failures.append(str(error))
    frame_count = args.rows * args.columns
    expected_size = (args.columns * PREP.CELL_WIDTH, args.rows * PREP.CELL_HEIGHT)
    report = read_prepare_report(args.prepare_report, failures)
    report_contract = validate_report_contract(
        report,
        storyboard=args.storyboard,
        neutral=args.neutral,
        atlas=args.atlas,
        rows=args.rows,
        columns=args.columns,
        failures=failures,
    )

    source_analysis = None
    try:
        source_analysis = PREP.analyze_storyboard(args.storyboard, rows=args.rows, columns=args.columns)
    except Exception as error:
        failures.append(f"source_storyboard_rejected:{error}")

    neutral_image: Image.Image | None = None
    neutral_metrics: dict[str, Any] | None = None
    try:
        neutral_image, neutral_metrics = PREP.require_valid_neutral(args.neutral)
    except Exception as error:
        failures.append(f"neutral_rejected:{error}")

    atlas_image: Image.Image | None = None
    atlas_mode: str | None = None
    try:
        raw_atlas = Image.open(args.atlas)
        atlas_mode = raw_atlas.mode
        if raw_atlas.mode != "RGBA":
            failures.append(f"atlas_mode={raw_atlas.mode}; expected RGBA")
        atlas_image = raw_atlas.convert("RGBA")
        if atlas_image.size != expected_size:
            failures.append(f"atlas_size={atlas_image.size}; expected {expected_size}")
    except OSError as error:
        failures.append(f"atlas_unreadable:{error}")

    frames: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    frame_repetitions: list[dict[str, Any]] = []
    if atlas_image is not None and atlas_image.size == expected_size:
        frames, masks = validate_atlas_cells(
            atlas_image,
            rows=args.rows,
            columns=args.columns,
            failures=failures,
        )
        if neutral_image is not None:
            neutral_array = np.asarray(neutral_image.convert("RGBA"), dtype=np.uint8)
            first = np.asarray(crop_cell(atlas_image, 0, columns=args.columns), dtype=np.uint8)
            final = np.asarray(crop_cell(atlas_image, frame_count - 1, columns=args.columns), dtype=np.uint8)
            if not np.array_equal(first, neutral_array):
                failures.append("F00_is_not_pixel_identical_to_runtime_neutral")
            if not np.array_equal(final, neutral_array):
                failures.append(f"F{frame_count - 1:02d}_is_not_pixel_identical_to_runtime_neutral")
            neutral_duplicates = [
                f"F{index:02d}"
                for index in range(1, frame_count - 1)
                if np.array_equal(np.asarray(crop_cell(atlas_image, index, columns=args.columns), dtype=np.uint8), neutral_array)
            ]
            if neutral_duplicates:
                failures.append("runtime_neutral_illegally_duplicated_in:" + ",".join(neutral_duplicates))
        frame_repetitions = validate_byte_identical_frame_repetitions(
            atlas_image,
            frame_count=frame_count,
            columns=args.columns,
            failures=failures,
        )
    else:
        failures.append("per_cell_validation_skipped_due_to_invalid_or_unreadable_atlas_geometry")

    continuity = continuity_metrics(frames, masks) if frames and masks else {
        "ok": False,
        "failures": ["continuity_unavailable_due_to_missing_valid_atlas_cells"],
    }
    failures.extend(continuity.get("failures", []))
    reconstruction = reconstruction_check(
        report,
        source_analysis,
        neutral_image,
        atlas_image if atlas_image is not None and atlas_image.size == expected_size else None,
        failures=failures,
    )

    validation = {
        "schema": "desktop-pet.interaction-v5.validation.v1",
        "ok": not failures,
        "storyboard": str(args.storyboard),
        "atlas": str(args.atlas),
        "prepare_report": str(args.prepare_report),
        "grid": {
            "rows": args.rows,
            "columns": args.columns,
            "frame_count": frame_count,
            "cell_size": [PREP.CELL_WIDTH, PREP.CELL_HEIGHT],
            "expected_atlas_size": list(expected_size),
            "supported_only_when_explicit": [
                {"rows": rows, "columns": columns, "frame_count": rows * columns}
                for rows, columns in sorted(PREP.SUPPORTED_GRIDS)
            ],
        },
        "atlas_mode": atlas_mode,
        "neutral_metrics": neutral_metrics,
        "prepare_report_contract": report_contract,
        "frames": frames,
        "byte_identical_frame_repetitions": frame_repetitions,
        "continuity": continuity,
        "reconstruction": reconstruction,
        "failures": failures,
        "repair_path": (
            "Keep the approved crouch/gaze assets untouched. Regenerate one single explicitly selected 8x8/64 or "
            "8x6/48 flat-magenta storyboard with one complete, connected cat in every slot, uniform identity and "
            "scale, gradual adjacent pose changes, and only the first/final exact neutral anchors. Then rerun prepare "
            "and this validator; do not patch faces/bodies or substitute independent strips."
        ),
    }
    write_json(args.json_out, validation)
    print(json.dumps({"ok": validation["ok"], "failure_count": len(failures), "json_out": str(args.json_out)}))
    raise SystemExit(0 if validation["ok"] else 1)


if __name__ == "__main__":
    main()
