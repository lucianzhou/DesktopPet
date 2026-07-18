#!/usr/bin/env python3
"""Register complete interaction cats by canonical head scale.

Every input is treated as one indivisible RGBA cat.  The only geometric
operation is a uniform whole-image resize followed by translation onto a
192x208 cell.  No body region is warped, composited, or imported from another
frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image


CELL_W = 192
CELL_H = 208
BASELINE = 204
VISIBLE_ALPHA = 8
REGIONS = {
    "head": (0.06, 0.43),
    "face": (0.18, 0.42),
    "chest": (0.48, 0.65),
}


def alpha_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def region(mask: np.ndarray, box: tuple[int, int, int, int], low: float, high: float) -> dict[str, float]:
    left, top, right, bottom = box
    height = bottom - top
    y0 = max(top, min(bottom, top + round(height * low)))
    y1 = max(y0, min(bottom, top + round(height * high)))
    spans: list[int] = []
    xs_all: list[np.ndarray] = []
    for y in range(y0, y1):
        xs = np.where(mask[y])[0]
        if len(xs):
            spans.append(int(xs.max()) - int(xs.min()) + 1)
            xs_all.append(xs)
    if not spans:
        raise ValueError("blank_anatomy_region")
    return {
        "outer_span": float(median(spans)),
        "center_x": float(np.concatenate(xs_all).mean()),
    }


def anatomy(image: Image.Image) -> dict[str, Any]:
    mask = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3] > VISIBLE_ALPHA
    box = alpha_box(mask)
    if box is None:
        raise ValueError("blank_frame")
    values = {name: region(mask, box, *bounds) for name, bounds in REGIONS.items()}
    # The face/head axis dominates; chest contributes enough to stop a tail or
    # asymmetric stance from horizontally recentering the complete cat.
    stable_axis = values["head"]["center_x"] * 0.65 + values["chest"]["center_x"] * 0.35
    return {"box": list(box), "stable_axis_x": stable_axis, **values}


def resize_premultiplied(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3]
    premultiplied = np.rint(rgba[..., :3] * alpha[..., None] / 255.0).astype(np.uint8)
    rgb = np.asarray(Image.fromarray(premultiplied, "RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    out_alpha = np.asarray(
        Image.fromarray(alpha.astype(np.uint8), "L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32
    )
    output = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    visible = out_alpha > 0
    output[..., 3] = np.rint(out_alpha).astype(np.uint8)
    output[..., :3][visible] = np.rint(
        np.clip(rgb[visible] * 255.0 / out_alpha[visible, None], 0, 255)
    ).astype(np.uint8)
    output[output[..., 3] <= VISIBLE_ALPHA] = 0
    return Image.fromarray(output, "RGBA")


def load_frames(repeated: list[Path], manifest: Path | None) -> list[tuple[str, Path]]:
    if repeated and manifest:
        raise ValueError("use_either_frame_alpha_or_manifest")
    if repeated:
        return [(f"F{index:02d}", path) for index, path in enumerate(repeated)]
    if manifest is None:
        raise ValueError("at_least_one_frame_is_required")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    items = raw.get("frames", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("manifest_frames_must_be_a_list")
    result: list[tuple[str, Path]] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            name, value = f"F{index:02d}", item
        elif isinstance(item, dict):
            name = str(item.get("name", item.get("frame", f"F{index:02d}")))
            value = item.get("path", item.get("frame_alpha"))
            if value is None:
                raise ValueError(f"manifest_frame_{index}_has_no_path")
        else:
            raise ValueError(f"invalid_manifest_frame_{index}")
        path = Path(value)
        if not path.is_absolute():
            path = manifest.parent / path
        result.append((name, path))
    return result


def register_frame(image: Image.Image, canonical: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    source = anatomy(image)
    source_box = tuple(source["box"])
    crop = image.crop(source_box)
    ratios = [
        canonical["head"]["outer_span"] / source["head"]["outer_span"],
        canonical["face"]["outer_span"] / source["face"]["outer_span"],
    ]
    requested_scale = float(median(ratios))
    fit_scale = min(CELL_W / crop.width, BASELINE / crop.height)
    base_scale = min(requested_scale, fit_scale)

    # Probe nearby integer raster sizes and choose the best simultaneous
    # head/face match.  Each probe remains one rigid, uniform whole-cat resize.
    candidates: list[tuple[tuple[float, float, float], Image.Image, dict[str, Any], float]] = []
    ideal_height = crop.height * base_scale
    for height in sorted({max(1, round(ideal_height) + delta) for delta in range(-2, 3)}):
        scale = height / crop.height
        if scale > fit_scale + 1e-12:
            continue
        width = max(1, round(crop.width * scale))
        if width > CELL_W:
            continue
        resized = resize_premultiplied(crop, (width, height))
        measured = anatomy(resized)
        error = (
            abs(measured["head"]["outer_span"] - canonical["head"]["outer_span"])
            + abs(measured["face"]["outer_span"] - canonical["face"]["outer_span"])
        )
        score = (error, abs(scale - requested_scale), -scale)
        candidates.append((score, resized, measured, scale))
    if not candidates:
        raise ValueError("head_lock_has_no_nonclipping_scale")
    _, resized, resized_metrics, scale = min(candidates, key=lambda item: item[0])
    resized_box = tuple(resized_metrics["box"])

    desired_x = round(canonical["stable_axis_x"] - resized_metrics["stable_axis_x"])
    minimum_x = -resized_box[0]
    maximum_x = CELL_W - resized_box[2]
    x = min(max(desired_x, minimum_x), maximum_x)
    y = BASELINE - resized_box[3]
    output_box = [
        x + resized_box[0], y + resized_box[1], x + resized_box[2], y + resized_box[3]
    ]
    clipped = output_box[0] < 0 or output_box[1] < 0 or output_box[2] > CELL_W or output_box[3] > CELL_H
    if clipped:
        raise ValueError(f"head_lock_would_clip:{output_box}")
    cell = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
    cell.alpha_composite(resized, (x, y))
    final = anatomy(cell)
    return cell, {
        "source_bbox": list(source_box),
        "source_head_outer_span": source["head"]["outer_span"],
        "source_face_span": source["face"]["outer_span"],
        "target_head_outer_span": canonical["head"]["outer_span"],
        "target_face_span": canonical["face"]["outer_span"],
        "requested_scale": round(requested_scale, 8),
        "fit_scale_limit": round(fit_scale, 8),
        "fit_limited": requested_scale > fit_scale + 1e-12,
        "applied_scale": round(scale, 8),
        "output_head_outer_span": final["head"]["outer_span"],
        "output_face_span": final["face"]["outer_span"],
        "output_bbox": final["box"],
        "baseline": final["box"][3],
        "translation": [x, y],
        "stable_axis_target": round(canonical["stable_axis_x"], 5),
        "stable_axis_output": round(final["stable_axis_x"], 5),
        "clipped": clipped,
        "whole_cat_uniform_scale_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-lock and baseline-register complete transparent cats.")
    parser.add_argument("--frame-alpha", action="append", default=[], type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--canonical-cell", required=True, type=Path)
    parser.add_argument("--output-atlas", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_frames(args.frame_alpha, args.manifest)
    canonical_image = Image.open(args.canonical_cell).convert("RGBA")
    if canonical_image.size != (CELL_W, CELL_H):
        raise ValueError(f"canonical_cell_must_be_192x208_not_{canonical_image.size}")
    canonical = anatomy(canonical_image)
    atlas = Image.new("RGBA", (CELL_W * len(frames), CELL_H), (0, 0, 0, 0))
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (name, path) in enumerate(frames):
        cell, record = register_frame(Image.open(path).convert("RGBA"), canonical)
        output_path = args.frames_dir / f"{index:02d}-{name}.png"
        cell.save(output_path)
        atlas.alpha_composite(cell, (index * CELL_W, 0))
        records.append({"frame": name, "input": str(path), "output": str(output_path), **record})
    args.output_atlas.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(args.output_atlas)
    report = {
        "schema": "desktop-pet.interaction-sequence-head-locked.v1",
        "canonical_cell": str(args.canonical_cell),
        "canonical_head_outer_span": canonical["head"]["outer_span"],
        "canonical_face_span": canonical["face"]["outer_span"],
        "baseline": BASELINE,
        "cross_frame_blending": False,
        "local_warp_or_face_composite": False,
        "frames": records,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "frames": len(records), "atlas": str(args.output_atlas), "report": str(args.report)}))


if __name__ == "__main__":
    main()
