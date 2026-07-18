#!/usr/bin/env python3
"""Register a complete-cat wake sequence without blending or local warping.

The manifest groups generated poses by source strip. Every generated group is
fit with one shared uniform scale, so the relative proportions authored inside
that strip are preserved. Approved cell anchors may be copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


CELL = (192, 208)
CONTENT = (176, 196)
BASELINE = 204
VISIBLE_ALPHA = 8


def alpha_box(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    ys, xs = np.where(alpha > VISIBLE_ALPHA)
    if len(xs) == 0:
        raise ValueError("blank source frame")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_visible(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    return rgba.crop(alpha_box(rgba))


def resize_premultiplied(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    source = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = source[..., 3:4] / 255.0
    premultiplied = np.concatenate((source[..., :3] * alpha, source[..., 3:4]), axis=2)
    resized = np.asarray(
        Image.fromarray(np.clip(premultiplied, 0, 255).astype(np.uint8), "RGBA").resize(
            size, Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    out_alpha = resized[..., 3:4]
    rgb = np.zeros_like(resized[..., :3])
    np.divide(resized[..., :3] * 255.0, out_alpha, out=rgb, where=out_alpha > 0)
    result = np.concatenate((np.clip(rgb, 0, 255), np.clip(out_alpha, 0, 255)), axis=2).astype(np.uint8)
    result[result[..., 3] == 0, :3] = 0
    return Image.fromarray(result, "RGBA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    columns = int(manifest.get("columns", 8))
    if columns <= 0:
        raise ValueError("columns must be positive")
    groups = manifest["groups"]
    frames = manifest["frames"]
    if not frames:
        raise ValueError("manifest has no frames")

    resolved: list[tuple[dict[str, object], Path, Image.Image]] = []
    grouped_sizes: dict[str, list[tuple[int, int]]] = {}
    for entry in frames:
        source = (args.manifest.parent / str(entry["source"])).resolve()
        image = Image.open(source).convert("RGBA")
        group_name = str(entry["group"])
        mode = str(groups[group_name]["mode"])
        crop = image if mode == "copy" else crop_visible(image)
        resolved.append((entry, source, crop))
        if mode != "copy":
            grouped_sizes.setdefault(group_name, []).append(crop.size)

    group_scales: dict[str, float | None] = {}
    for name, settings in groups.items():
        mode = str(settings["mode"])
        if mode == "copy":
            group_scales[name] = None
            continue
        if mode == "fixed":
            scale = float(settings["scale"])
        elif mode == "fit":
            sizes = grouped_sizes[name]
            scale = min(
                CONTENT[0] / max(width for width, _ in sizes),
                CONTENT[1] / max(height for _, height in sizes),
            ) * float(settings.get("multiplier", 1.0))
        else:
            raise ValueError(f"unsupported group mode {mode!r}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid scale for group {name}: {scale}")
        group_scales[name] = scale

    rows = math.ceil(len(frames) / columns)
    atlas = Image.new("RGBA", (columns * CELL[0], rows * CELL[1]), (0, 0, 0, 0))
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, (entry, source, crop) in enumerate(resolved):
        group_name = str(entry["group"])
        mode = str(groups[group_name]["mode"])
        scale = group_scales[group_name]
        if mode == "copy":
            if crop.size != CELL:
                raise ValueError(f"copy frame {source} is {crop.size}, expected {CELL}")
            cell = crop.copy()
            operation = "exact_cell_copy"
        else:
            assert scale is not None
            target_size = (
                max(1, round(crop.width * scale)),
                max(1, round(crop.height * scale)),
            )
            resized = resize_premultiplied(crop, target_size)
            cell = Image.new("RGBA", CELL, (0, 0, 0, 0))
            x = (CELL[0] - target_size[0]) // 2 + int(entry.get("offset_x", 0))
            y = BASELINE - target_size[1] + int(entry.get("offset_y", 0))
            cell.alpha_composite(resized, (x, y))
            operation = "whole_cat_uniform_scale_and_translation"

        box = alpha_box(cell)
        safe_margin = int(groups[group_name].get("safe_margin", 8))
        if (
            safe_margin < 0
            or box[0] < safe_margin
            or box[2] > CELL[0] - safe_margin
            or box[1] < safe_margin
            or box[3] != BASELINE
        ):
            raise ValueError(f"frame {index} registered outside safe bounds: {box}")
        array = np.asarray(cell, dtype=np.uint8).copy()
        array[array[..., 3] == 0, :3] = 0
        cell = Image.fromarray(array, "RGBA")
        frame_path = args.frames_dir / f"{index:02d}.png"
        cell.save(frame_path)
        row, column = divmod(index, columns)
        atlas.alpha_composite(cell, (column * CELL[0], row * CELL[1]))
        records.append(
            {
                "frame": index,
                "label": entry.get("label", f"F{index:02d}"),
                "source": str(source),
                "group": group_name,
                "operation": operation,
                "group_scale": scale,
                "safe_margin": safe_margin,
                "source_size": list(crop.size),
                "registered_box": list(box),
                "visible_area": int((array[..., 3] > VISIBLE_ALPHA).sum()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(args.output)
    report = {
        "schema": "desktop-pet.baomihua-wake-registration.v2",
        "ok": True,
        "operation": "complete_cat_shared_group_scale_and_translation_only",
        "interpolation": False,
        "cross_frame_blending": False,
        "local_warping": False,
        "face_or_body_patching": False,
        "frame_count": len(frames),
        "columns": columns,
        "rows": rows,
        "group_scales": group_scales,
        "frames": records,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "frames": len(frames), "output": str(args.output)}))


if __name__ == "__main__":
    main()
