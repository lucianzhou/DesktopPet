#!/usr/bin/env python3
"""Strict QA and deterministic registration for generated gaze source strips.

This script is intentionally a gate, not an atlas assembler.  A source strip is
accepted only when it contains eight complete, disconnected cat silhouettes.
It produces a diagnostic contact sheet and a JSON report even when the source
fails, so a bad strip cannot silently turn into cropped or head/body-split
cells.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(__file__).resolve().with_name("prepare-canonical.py")
SPEC = importlib.util.spec_from_file_location("prepare_canonical", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load {HELPER_PATH}")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)

CELL_SIZE = tuple(HELPER.CELL_SIZE)
CONTENT_SIZE = tuple(HELPER.CONTENT_SIZE)
BASELINE_Y = int(HELPER.BASELINE_Y)

ROWS = ("row9", "row10")
ROW_DIRECTIONS = {
    "row9": ["000 up", "022.5 up-right", "045 up-right", "067.5 up-right", "090 right", "112.5 down-right", "135 down-right", "157.5 down-right"],
    "row10": ["180 down", "202.5 down-left", "225 down-left", "247.5 down-left", "270 left", "292.5 up-left", "315 up-left", "337.5 up-left"],
}
COMPONENT_MIN_PIXELS = 500
VISIBLE_ALPHA = 8


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return all 8-connected components in deterministic left-to-right order."""

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        points: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(points) >= COMPONENT_MIN_PIXELS:
            components.append(points)
    components.sort(key=lambda points: (min(x for _, x in points), min(y for y, _ in points)))
    return components


def component_box(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    points = list(points)
    ys = [y for y, _ in points]
    xs = [x for _, x in points]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def crop_component(rgba: Image.Image, points: list[tuple[int, int]]) -> Image.Image:
    """Crop only one component and clear all hidden RGB."""

    arr = np.asarray(rgba.convert("RGBA"), dtype=np.uint8)
    left, top, right, bottom = component_box(points)
    crop = arr[top:bottom, left:right].copy()
    keep = np.zeros(crop.shape[:2], dtype=bool)
    for y, x in points:
        keep[y - top, x - left] = True
    crop[~keep, 3] = 0
    crop[crop[..., 3] == 0, :3] = 0
    return Image.fromarray(crop, "RGBA")


def register(crop: Image.Image, scale: float) -> Image.Image:
    width, height = crop.size
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = crop.resize(size, Image.Resampling.LANCZOS)
    cell = Image.new("RGBA", CELL_SIZE, (0, 0, 0, 0))
    x = (CELL_SIZE[0] - size[0]) // 2
    y = BASELINE_Y - size[1]
    cell.alpha_composite(resized, (x, y))
    arr = np.asarray(cell, dtype=np.uint8).copy()
    arr[arr[..., 3] == 0, :3] = 0
    return Image.fromarray(arr, "RGBA")


def body_mask(image: Image.Image, y_start: int = 70) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"))
    mask = arr[..., 3] > VISIBLE_ALPHA
    mask[:y_start] = False
    return mask


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Visible pixels touching transparent space in an 8-neighbor ring."""

    outside = ~mask
    near_outside = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            near_outside |= np.roll(np.roll(outside, dy, axis=0), dx, axis=1)
    return mask & near_outside


def pink_edge_pixels(image: Image.Image) -> int:
    """Count likely key-colored pixels left on a silhouette boundary.

    The source key is strongly magenta, while Baomihua intentionally has a
    pink nose and pink inner-ear fur that can legitimately reach a silhouette
    edge.  A score of 25 is high enough to catch key-colored RGB bleed without
    misclassifying those real identity details as a chroma failure.
    """

    arr = np.asarray(image.convert("RGBA"), dtype=np.int16)
    alpha = arr[..., 3]
    mask = alpha > VISIBLE_ALPHA
    rgb = arr[..., :3]
    score = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
    edge = boundary_mask(mask)
    return int(np.count_nonzero(edge & (score >= 25) & (alpha >= 180)))


def canonical_body(canonical: Image.Image) -> np.ndarray:
    # The approved canonical cell is already registered.  A source canonical
    # image is also accepted for convenience by running the shared key removal
    # and registration path.
    if canonical.size == CELL_SIZE and canonical.mode == "RGBA":
        return body_mask(canonical)
    keyed = HELPER.remove_magenta_key(canonical.convert("RGBA"))
    cell, _ = HELPER.register_cell(keyed)
    return body_mask(cell)


def body_metrics(cell: Image.Image, reference: np.ndarray) -> dict[str, float | int]:
    mask = body_mask(cell)
    inter = int(np.logical_and(mask, reference).sum())
    union = int(np.logical_or(mask, reference).sum())
    area = int(mask.sum())
    ref_area = int(reference.sum())
    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return {
        "body_iou": round(inter / union, 5) if union else 0.0,
        "body_area_ratio": round(area / ref_area, 5) if ref_area else 0.0,
        "registered_box": list(box),
        "bottom_y": box[3],
    }


def diagnostics_for_row(path: Path, row_name: str, canonical_ref: np.ndarray) -> tuple[dict, list[Image.Image]]:
    original = Image.open(path).convert("RGBA")
    keyed = HELPER.remove_magenta_key(original)
    rgba = np.asarray(keyed)
    mask = rgba[..., 3] > VISIBLE_ALPHA
    height, width = mask.shape
    components = connected_components(mask)
    boxes = [component_box(points) for points in components]

    reasons: list[str] = []
    if len(components) != 8:
        reasons.append(f"expected_8_complete_components_found_{len(components)}")

    source_edge_components = []
    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        if left <= 0 or top <= 0 or right >= width or bottom >= height:
            source_edge_components.append(index)
    if source_edge_components:
        reasons.append("silhouette_touches_source_edge:" + ",".join(map(str, source_edge_components)))

    # A component's center must be inside its expected eight-way slot.  This
    # catches a montage that has lost a frame or has two poses merged into one.
    slot_mismatches = []
    for index, box in enumerate(boxes):
        center = (box[0] + box[2]) / 2
        slot_left = index * width / 8
        slot_right = (index + 1) * width / 8
        if not (slot_left - width * 0.06 <= center <= slot_right + width * 0.06):
            slot_mismatches.append(index)
    if slot_mismatches:
        reasons.append("component_center_outside_expected_slot:" + ",".join(map(str, slot_mismatches)))

    candidate_cells: list[Image.Image] = []
    metrics: list[dict] = []
    if len(components) == 8:
        heights = [box[3] - box[1] for box in boxes]
        widths = [box[2] - box[0] for box in boxes]
        max_width = max(widths)
        max_height = max(heights)
        scale = min(CONTENT_SIZE[0] / max_width, CONTENT_SIZE[1] / max_height)
        if max_width * scale > CONTENT_SIZE[0] or max_height * scale > CONTENT_SIZE[1]:
            reasons.append("registered_silhouette_exceeds_content_limit")
        for index, points in enumerate(components):
            cell = register(crop_component(keyed, points), scale)
            candidate_cells.append(cell)
            item = {
                "index": index,
                "source_box": list(boxes[index]),
                "source_width": widths[index],
                "source_height": heights[index],
                "pink_edge_pixels": pink_edge_pixels(cell),
            }
            item.update(body_metrics(cell, canonical_ref))
            metrics.append(item)
        bad_pink = [item["index"] for item in metrics if item["pink_edge_pixels"] > 0]
        if bad_pink:
            reasons.append("key_color_edge_spill:" + ",".join(map(str, bad_pink)))
        bad_body = [
            item["index"]
            for item in metrics
            if item["body_iou"] < 0.82
            or not 0.82 <= item["body_area_ratio"] <= 1.18
            or item["bottom_y"] != BASELINE_Y
        ]
        if bad_body:
            reasons.append("body_registration_or_identity_drift:" + ",".join(map(str, bad_body)))
        if metrics:
            areas = [float(item["body_area_ratio"]) for item in metrics]
            if max(areas) - min(areas) > 0.12:
                reasons.append("adjacent_body_area_variation_exceeds_12_percent")

    accepted = not reasons
    report = {
        "source": str(path),
        "source_size": [width, height],
        "expected_frames": 8,
        "directions": ROW_DIRECTIONS[row_name],
        "component_count": len(components),
        "accepted": accepted,
        "reasons": reasons,
        "components": metrics,
        "note": (
            "Rejected source strips are never safe to assemble. Regenerate the row as one "
            "coherent 8-pose strip with a visible horizontal gutter and all silhouettes fully "
            "inside the image; rerun this gate before extraction."
            if not accepted
            else "Source passed strict component, registration, body, and edge-spill gates."
        ),
    }
    return report, candidate_cells


def draw_contact(output: Path, rows: list[tuple[str, dict, list[Image.Image]]]) -> None:
    tile_w, tile_h = 192, 232
    width = tile_w * 8
    height = tile_h * len(rows)
    sheet = Image.new("RGB", (width, height), (52, 52, 55))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = None
    for row_index, (name, report, cells) in enumerate(rows):
        y0 = row_index * tile_h
        title = f"{name}: {'PASS' if report['accepted'] else 'FAIL'}"
        draw.text((4, y0 + 3), title, fill=(255, 255, 255), font=font)
        for index in range(8):
            x0 = index * tile_w
            draw.rectangle((x0, y0 + 20, x0 + tile_w - 1, y0 + tile_h - 1), outline=(100, 100, 105))
            if index < len(cells):
                # Use a dark background and a red edge marker for residual
                # magenta; this makes halos visible in review.
                cell = cells[index].copy()
                bg = Image.new("RGBA", cell.size, (52, 52, 55, 255))
                bg.alpha_composite(cell)
                sheet.paste(bg.convert("RGB"), (x0, y0 + 20))
                if report["components"][index]["pink_edge_pixels"] > 0:
                    draw.rectangle((x0 + 1, y0 + 21, x0 + tile_w - 2, y0 + tile_h - 2), outline=(235, 70, 70), width=2)
            elif not report["accepted"]:
                draw.text((x0 + 67, y0 + 116), "NO CELL", fill=(255, 90, 90), font=font)
            draw.text((x0 + 5, y0 + 27), f"{index:02d}", fill=(255, 255, 255), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row9", required=True, type=Path)
    parser.add_argument("--row10", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    canonical = Image.open(args.canonical).convert("RGBA")
    canonical_ref = canonical_body(canonical)
    rows: list[tuple[str, dict, list[Image.Image]]] = []
    for name, path in (("row9", args.row9), ("row10", args.row10)):
        report, cells = diagnostics_for_row(path, name, canonical_ref)
        rows.append((name, report, cells))
        # Keep deterministic candidates beside the report for human review.
        # These are never treated as production assets: a row remains rejected
        # until every gate below passes.
        candidate_dir = args.output_dir / "candidates" / name
        if cells:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for index, cell in enumerate(cells):
                cell.save(candidate_dir / f"{index:02d}.png")

    report = {
        "schema": "desktop-pet.gaze-source-qa.v1",
        "canonical": str(args.canonical),
        "accepted": all(row[1]["accepted"] for row in rows),
        "rows": {name: row_report for name, row_report, _ in rows},
        "repair_path": (
            "Regenerate both row strips with eight separate complete silhouettes, at least 24px "
            "horizontal gutter, no silhouette touching the source image edge, and no pink/key "
            "outline. Then rerun this script; only accepted candidate cells may be assembled."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gaze-source-v1-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    draw_contact(args.output_dir / "gaze-source-v1-contact.png", rows)
    print(json.dumps({"accepted": report["accepted"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
