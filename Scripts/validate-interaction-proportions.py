#!/usr/bin/env python3
"""Production QA gate for Baomihua interaction-frame proportions.

The interaction atlas is allowed to change pose, but it must never quietly
change into a thinner cat, a larger head, different paws, a shifted tail, or a
different camera.  This validator combines three kinds of evidence:

* registered 192x208 alpha geometry for the whole cat and its body regions;
* explicit C/H/S anchor interpolation for generated rise/return segments; and
* adjacent plus three-frame continuity checks for all body-bearing landmarks.

``--mode-map`` remains backwards compatible with the former simple mapping:

    {"frames": {"F00": "seated", "F01": "transition"}}

Production maps may instead supply a phase target per transition frame:

    {"frames": [
      {"frame": "F01", "mode": "transition", "anchor_pair": "C_H", "t": 0.16},
      {"frame": "F02", "mode": "transition", "anchor_pair": "C_H", "t": 0.32}
    ]}

``C`` is ``--canonical``, ``H`` is ``--half-rise-master``, and ``S`` is
``--standing-master``.  Phase targets are intentionally optional for legacy
maps, but a new generated C->H or H->S row should always declare them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image


CELL_W = 192
CELL_H = 208
VISIBLE_ALPHA = 8
BASELINE = 204
MODES = {"seated", "transition", "compact-standing"}
ANCHOR_PAIRS = {"C_H", "H_S"}
SYMMETRY_LIMITS = {
    "paws": 1.15,
    "legs": 1.20,
    "forelimb_proxy": 1.15,
}

# All body regions are deliberately expressed in a frame's own registered
# silhouette height.  This lets a small natural head tilt through while still
# measuring the same anatomical zones after a sit-to-stand pose change.
REGIONS = {
    "head": (0.06, 0.43),
    "face": (0.18, 0.42),
    "shoulder": (0.40, 0.57),
    "chest": (0.48, 0.65),
    "belly": (0.60, 0.78),
    "lower": (0.70, 0.89),
}

# Paths are scalar leaf values in the metrics document.  They are compared to
# the appropriate C/H/S phase target whenever an anchor_pair+t is declared.
TARGET_PATHS = (
    "width",
    "height",
    "area",
    "aspect",
    "center_x",
    "alpha_centroid_x",
    "alpha_centroid_y",
    "head.width",
    "head.height",
    "head.area",
    "head.outer_span",
    "head.center_x",
    "head.center_y",
    "face.outer_span",
    "face.area",
    "ears.left.width",
    "ears.left.height",
    "ears.left.area",
    "ears.left.tip_x",
    "ears.left.tip_y",
    "ears.left.outer_x",
    "ears.right.width",
    "ears.right.height",
    "ears.right.area",
    "ears.right.tip_x",
    "ears.right.tip_y",
    "ears.right.outer_x",
    "shoulder.outer_span",
    "shoulder.area",
    "chest.outer_span",
    "chest.area",
    "belly.outer_span",
    "belly.area",
    "lower.outer_span",
    "lower.area",
    "ratios.head_to_body_area",
    "ratios.shoulder_to_head",
    "ratios.chest_to_head",
    "ratios.belly_to_head",
    "ratios.lower_to_head",
    "ratios.forelimb_left_to_head",
    "ratios.forelimb_right_to_head",
    "legs.left.width",
    "legs.right.width",
    "legs.left.area",
    "legs.right.area",
    "legs.left.center_x",
    "legs.right.center_x",
    "legs.spacing",
    "legs.left.bottom",
    "legs.right.bottom",
    "forelimb_proxy.left.width",
    "forelimb_proxy.right.width",
    "forelimb_proxy.left.area",
    "forelimb_proxy.right.area",
    "forelimb_proxy.left.center_x",
    "forelimb_proxy.right.center_x",
    "forelimb_proxy.left.top",
    "forelimb_proxy.right.top",
    "forelimb_proxy.left.bottom",
    "forelimb_proxy.right.bottom",
    "forelimb_proxy.left.length",
    "forelimb_proxy.right.length",
    "paws.left.width",
    "paws.right.width",
    "paws.left.area",
    "paws.right.area",
    "paws.left.center_x",
    "paws.right.center_x",
    "paws.spacing",
    "paws.left.bottom",
    "paws.right.bottom",
    "tail.area",
    "tail.outer_span",
    "tail.root_x",
    "tail.root_y",
    "tail.left.area",
    "tail.left.outer_span",
    "tail.left.root_x",
    "tail.left.root_y",
    "tail.right.area",
    "tail.right.outer_span",
    "tail.right.root_x",
    "tail.right.root_y",
)

# A smaller list keeps continuity reports readable while covering every body
# part that can make the animation feel like it swaps to another cat.
TEMPORAL_PATHS = (
    "width",
    "height",
    "area",
    "aspect",
    "center_x",
    "alpha_centroid_x",
    "alpha_centroid_y",
    "head.width",
    "head.height",
    "head.area",
    "head.outer_span",
    "head.center_x",
    "head.center_y",
    "face.outer_span",
    "face.area",
    "ears.left.width",
    "ears.left.height",
    "ears.left.area",
    "ears.left.tip_x",
    "ears.left.tip_y",
    "ears.left.outer_x",
    "ears.right.width",
    "ears.right.height",
    "ears.right.area",
    "ears.right.tip_x",
    "ears.right.tip_y",
    "ears.right.outer_x",
    "shoulder.outer_span",
    "shoulder.area",
    "chest.outer_span",
    "chest.area",
    "belly.outer_span",
    "belly.area",
    "lower.outer_span",
    "lower.area",
    "legs.left.width",
    "legs.right.width",
    "legs.left.area",
    "legs.right.area",
    "legs.spacing",
    "forelimb_proxy.left.width",
    "forelimb_proxy.right.width",
    "forelimb_proxy.left.area",
    "forelimb_proxy.right.area",
    "forelimb_proxy.left.center_x",
    "forelimb_proxy.right.center_x",
    "forelimb_proxy.left.top",
    "forelimb_proxy.right.top",
    "forelimb_proxy.left.bottom",
    "forelimb_proxy.right.bottom",
    "forelimb_proxy.left.length",
    "forelimb_proxy.right.length",
    "paws.left.width",
    "paws.right.width",
    "paws.left.area",
    "paws.right.area",
    "paws.left.center_x",
    "paws.right.center_x",
    "paws.left.top",
    "paws.right.top",
    "paws.left.bottom",
    "paws.right.bottom",
    "paws.spacing",
    "ratios.head_to_body_area",
    "ratios.shoulder_to_head",
    "ratios.chest_to_head",
    "ratios.belly_to_head",
    "ratios.lower_to_head",
    "ratios.forelimb_left_to_head",
    "ratios.forelimb_right_to_head",
    "ratios.left_paw_to_head",
    "ratios.right_paw_to_head",
    "ratios.paw_spacing_to_head",
    "tail.area",
    "tail.outer_span",
    "tail.root_x",
    "tail.root_y",
    "tail.left.area",
    "tail.left.root_x",
    "tail.left.root_y",
    "tail.right.area",
    "tail.right.root_x",
    "tail.right.root_y",
)

# Identity-bearing values that must not briefly reverse direction inside a
# transition.  Large pose dimensions may change monotonically; this list
# catches a one-frame shrink followed by a rebound (or the inverse).
REVERSAL_PATHS = (
    "alpha_centroid_x",
    "alpha_centroid_y",
    "head.outer_span",
    "head.height",
    "face.outer_span",
    "ears.left.width",
    "ears.left.height",
    "ears.right.width",
    "ears.right.height",
    "forelimb_proxy.left.width",
    "forelimb_proxy.right.width",
    "forelimb_proxy.left.length",
    "forelimb_proxy.right.length",
    "paws.left.width",
    "paws.right.width",
    "paws.left.center_x",
    "paws.right.center_x",
    "paws.spacing",
    "ratios.head_to_body_area",
    "ratios.shoulder_to_head",
    "ratios.chest_to_head",
    "ratios.belly_to_head",
    "ratios.lower_to_head",
    "ratios.forelimb_left_to_head",
    "ratios.forelimb_right_to_head",
    "ratios.left_paw_to_head",
    "ratios.right_paw_to_head",
)

# A phase may contain small secondary motion, but its principal proportions
# must keep converging on the declared endpoint anchor as t advances.  These
# paths are deliberately human-readable in QA reports.
ANCHOR_PROGRESS_PATHS = (
    "width",
    "height",
    "area",
    "head.width",
    "head.height",
    "head.area",
    "shoulder.outer_span",
    "chest.outer_span",
    "belly.outer_span",
    "paws.left.width",
    "paws.right.width",
    "paws.left.area",
    "paws.right.area",
    "paws.spacing",
    "tail.outer_span",
    "tail.left.area",
    "tail.right.area",
)


def alpha_mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3] > VISIBLE_ALPHA


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Return true [start, end) intervals in a one-dimensional alpha row."""

    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(row.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def alpha_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def connected_component_count(mask: np.ndarray) -> tuple[int, list[int]]:
    """Count all 8-connected alpha components, including tiny stray islands."""

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        seen[y, x] = True
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        sizes.append(size)
    return len(sizes), sorted(sizes, reverse=True)


def region_metrics(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    low: float,
    high: float,
) -> dict[str, Any]:
    """Measure one anatomy band using both full-span and alpha-area evidence."""

    left, top, right, bottom = box
    height = bottom - top
    y0 = max(top, min(bottom, top + round(height * low)))
    y1 = max(y0, min(bottom, top + round(height * high)))
    submask = mask[y0:y1]
    ys, xs = np.where(submask)
    if not len(xs):
        return {
            "box": [0, 0, 0, 0],
            "width": 0.0,
            "height": 0.0,
            "area": 0.0,
            "outer_span": 0.0,
            "central_span": 0.0,
            "center_x": None,
            "center_y": None,
            "samples": 0,
        }
    absolute_y = ys + y0
    region_box = (int(xs.min()), int(absolute_y.min()), int(xs.max()) + 1, int(absolute_y.max()) + 1)
    center_x = (left + right - 1) / 2.0
    outer_spans: list[int] = []
    central_spans: list[int] = []
    for y in range(y0, y1):
        row_x = np.where(mask[y])[0]
        if not len(row_x):
            continue
        outer_spans.append(int(row_x.max()) - int(row_x.min()) + 1)
        row_runs = runs(mask[y])
        containing = [item for item in row_runs if item[0] <= center_x < item[1]]
        chosen = max(containing or row_runs, key=lambda item: item[1] - item[0])
        central_spans.append(chosen[1] - chosen[0])
    return {
        "box": list(region_box),
        "width": float(region_box[2] - region_box[0]),
        "height": float(region_box[3] - region_box[1]),
        "area": float(len(xs)),
        "outer_span": float(median(outer_spans)) if outer_spans else 0.0,
        "central_span": float(median(central_spans)) if central_spans else 0.0,
        "center_x": round(float(xs.mean()), 5),
        "center_y": round(float(absolute_y.mean()), 5),
        "samples": len(outer_spans),
    }


def ear_silhouette_metrics(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    head_center_x: float,
) -> dict[str, Any]:
    """Measure stable left/right upper-head silhouette proxies for the ears.

    Alpha alone cannot semantically segment fur from an attached ear, so this
    intentionally measures each half of the top 28% of the registered cat.
    Tip and outer-edge motion still catches the visible ear size/placement pop
    without requiring generated part masks.
    """

    left, top, right, bottom = box
    upper_bottom = min(bottom, top + max(8, round((bottom - top) * 0.28)))
    split = int(round(head_center_x))

    def side(name: str) -> dict[str, Any]:
        x0, x1 = (left, min(right, split + 1)) if name == "left" else (max(left, split), right)
        submask = mask[top:upper_bottom, x0:x1]
        ys, xs = np.where(submask)
        if not len(xs):
            return {
                "width": None,
                "height": None,
                "area": None,
                "center_x": None,
                "center_y": None,
                "tip_x": None,
                "tip_y": None,
                "outer_x": None,
            }
        absolute_x = xs + x0
        absolute_y = ys + top
        tip_y = int(absolute_y.min())
        tip_band = absolute_x[absolute_y <= tip_y + 1]
        return {
            "width": float(absolute_x.max() - absolute_x.min() + 1),
            "height": float(absolute_y.max() - absolute_y.min() + 1),
            "area": float(len(absolute_x)),
            "center_x": round(float(absolute_x.mean()), 5),
            "center_y": round(float(absolute_y.mean()), 5),
            "tip_x": round(float(np.median(tip_band)), 5),
            "tip_y": float(tip_y),
            "outer_x": float(absolute_x.min() if name == "left" else absolute_x.max()),
        }

    return {
        "method": "upper_head_half_silhouette_v1",
        "band": [top, upper_bottom],
        "left": side("left"),
        "right": side("right"),
    }


def empty_support_side() -> dict[str, Any]:
    return {
        "width": None,
        "area": None,
        "center_x": None,
        "center_y": None,
        "top": None,
        "bottom": None,
        "samples": 0,
    }


def choose_support_run(
    choices: Iterable[tuple[int, int]],
    *,
    side: str,
    center_x: float,
    head_width: float,
    target_offset_ratio: float,
) -> tuple[int, int] | None:
    """Prefer a central paw/leg over a wider lateral tail run."""

    # A continuous belly/forechest run crossing centre is not a left or right
    # support.  Rejecting it here is what prevents a seated body mass from
    # being reported as a 120px-wide paw.
    candidates = [
        item
        for item in choices
        if item[1] - item[0] >= 3
        and (item[1] <= center_x + 1.0 if side == "left" else item[0] >= center_x - 1.0)
    ]
    if not candidates:
        return None
    target = center_x + (-1.0 if side == "left" else 1.0) * head_width * target_offset_ratio
    return min(
        candidates,
        key=lambda item: (abs((item[0] + item[1]) / 2.0 - target), -(item[1] - item[0])),
    )


def support_metrics(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    head_width: float,
    body_center_x: float,
    *,
    top_offset: int,
    bottom_offset: int,
    allow_center_split: bool = False,
    window_scale: float = 0.78,
    target_offset_ratio: float = 0.24,
    target_clip_ratio: float | None = None,
    minimum_samples: int = 8,
) -> dict[str, Any]:
    """Measure left/right legs or paws independently across a lower-body band."""

    left, _top, right, bottom = box
    # The silhouette bbox is biased by Baomihua's lateral tail.  Head centre
    # is the stable anatomical axis for locating the two front supports.
    center_x = body_center_x
    half_window = head_width * window_scale
    window_left = max(left, round(center_x - half_window))
    window_right = min(right, round(center_x + half_window))
    samples: dict[str, list[tuple[int, int, float]]] = {"left": [], "right": []}
    pairs: list[tuple[int, int]] = []
    y0 = max(0, bottom - top_offset)
    y1 = max(y0, bottom - bottom_offset)
    for y in range(y0, y1):
        row_runs = [(window_left + start, window_left + end) for start, end in runs(mask[y, window_left:window_right])]
        # Adjacent front paws can touch at the bottom.  For the paw band we
        # retain their individual floor contacts by splitting that one joined
        # central run at the registered body centre; legs never use this path.
        if allow_center_split:
            expanded: list[tuple[int, int]] = []
            split_at = int(round(center_x))
            for item in row_runs:
                if item[0] < split_at < item[1] and item[1] - item[0] >= 8:
                    expanded.extend([(item[0], split_at), (split_at, item[1])])
                else:
                    expanded.append(item)
            row_runs = expanded
        if target_clip_ratio is not None:
            clipped: list[tuple[int, int]] = []
            clip_half = max(4.0, head_width * target_clip_ratio)
            for side in ("left", "right"):
                target = center_x + (-1.0 if side == "left" else 1.0) * head_width * target_offset_ratio
                clip_left = int(math.floor(target - clip_half))
                clip_right = int(math.ceil(target + clip_half))
                for start, end in row_runs:
                    candidate = (max(start, clip_left), min(end, clip_right))
                    if candidate[1] - candidate[0] >= 3:
                        clipped.append(candidate)
            row_runs = clipped
        selected = {
            side: choose_support_run(
                row_runs,
                side=side,
                center_x=center_x,
                head_width=head_width,
                target_offset_ratio=target_offset_ratio,
            )
            for side in ("left", "right")
        }
        if selected["left"] is not None and selected["right"] is not None:
            for side, item in selected.items():
                assert item is not None
                samples[side].append((y, item[1] - item[0], (item[0] + item[1]) / 2.0))
            pairs.append((selected["left"][1] - selected["left"][0], selected["right"][1] - selected["right"][0]))

    def summarise(values: list[tuple[int, int, float]]) -> dict[str, Any]:
        if not values:
            return empty_support_side()
        ys = [item[0] for item in values]
        widths = [item[1] for item in values]
        centers = [item[2] for item in values]
        return {
            "width": float(median(widths)),
            "area": float(sum(widths)),
            "center_x": round(float(median(centers)), 5),
            "center_y": round(float(median(ys)), 5),
            "top": int(min(ys)),
            "bottom": int(max(ys) + 1),
            "samples": len(values),
        }

    left_side, right_side = summarise(samples["left"]), summarise(samples["right"])
    combined = None
    ratio = None
    spacing = None
    if pairs:
        combined = float(median([a + b for a, b in pairs]))
    if left_side["width"] is not None and right_side["width"] is not None:
        ratio = max(left_side["width"], right_side["width"]) / max(1.0, min(left_side["width"], right_side["width"]))
        spacing = abs(right_side["center_x"] - left_side["center_x"])
    coverage = len(pairs) / max(1, y1 - y0)
    return {
        "left": left_side,
        "right": right_side,
        "combined": combined,
        "ratio": ratio,
        "spacing": spacing,
        "samples": len(pairs),
        "coverage": round(coverage, 6),
        "reliable": len(pairs) >= minimum_samples,
        # Legacy aliases retained for scripts/report readers that consumed the
        # first version of this validator.
        "left_width": left_side["width"],
        "right_width": right_side["width"],
    }


def forelimb_column_metrics(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    head_width: float,
    body_center_x: float,
) -> dict[str, Any]:
    """Track both front-support columns even while they merge into the torso.

    The stricter run-based leg tracker is preferable once two legs separate,
    but seated/rising cats often form one continuous alpha mass.  Fixed
    anatomical windows around the two approved paw axes provide a continuous
    width/area/visible-length proxy instead of silently skipping those frames.
    """

    left, _top, right, bottom = box
    y0 = max(0, bottom - 64)
    split_gap = max(1, round(head_width * 0.02))
    outer_reach = max(8, round(head_width * 0.50))

    def side(name: str) -> dict[str, Any]:
        if name == "left":
            x0 = max(left, round(body_center_x - outer_reach))
            x1 = min(right, round(body_center_x - split_gap))
        else:
            x0 = max(left, round(body_center_x + split_gap))
            x1 = min(right, round(body_center_x + outer_reach))
        submask = mask[y0:bottom, x0:x1]
        ys, xs = np.where(submask)
        if not len(xs):
            return {
                "width": None,
                "area": None,
                "center_x": None,
                "center_y": None,
                "top": None,
                "bottom": None,
                "length": None,
                "samples": 0,
            }
        absolute_x = xs + x0
        absolute_y = ys + y0
        scan_widths = [int(mask[y, x0:x1].sum()) for y in range(y0, bottom) if mask[y, x0:x1].any()]
        top_y, bottom_y = int(absolute_y.min()), int(absolute_y.max()) + 1
        return {
            "width": float(median(scan_widths)),
            "area": float(len(absolute_x)),
            "center_x": round(float(absolute_x.mean()), 5),
            "center_y": round(float(absolute_y.mean()), 5),
            "top": top_y,
            "bottom": bottom_y,
            "length": float(bottom_y - top_y),
            "samples": len(scan_widths),
        }

    left_side, right_side = side("left"), side("right")
    return {
        "method": "fixed_front_support_columns_v1",
        "left": left_side,
        "right": right_side,
        "ratio": ratio(left_side["width"], right_side["width"]),
    }


def tail_metrics(mask: np.ndarray, box: tuple[int, int, int, int], head_width: float) -> dict[str, Any]:
    """Approximate the lateral tail-root mass without confusing it with paws."""

    left, top, right, bottom = box
    height = bottom - top
    center_x = (left + right - 1) / 2.0
    y0 = max(top, top + round(height * 0.48))
    y1 = min(bottom, top + round(height * 0.90))
    core_half = max(head_width * 0.36, (right - left) * 0.25)
    core_left = center_x - core_half
    core_right = center_x + core_half

    def side(side: str) -> dict[str, Any]:
        ys_all: list[int] = []
        xs_all: list[int] = []
        spans: list[int] = []
        for y in range(y0, y1):
            xs = np.where(mask[y])[0]
            if side == "left":
                xs = xs[xs < core_left]
            else:
                xs = xs[xs > core_right]
            if not len(xs):
                continue
            ys_all.extend([y] * len(xs))
            xs_all.extend(int(item) for item in xs)
            spans.append(int(xs.max()) - int(xs.min()) + 1)
        if not xs_all:
            return {"area": 0.0, "outer_span": 0.0, "root_x": None, "root_y": None, "box": [0, 0, 0, 0]}
        xs_array = np.asarray(xs_all)
        ys_array = np.asarray(ys_all)
        if side == "left":
            boundary = float(np.quantile(xs_array, 0.92))
            root_mask = xs_array >= boundary - 1.0
        else:
            boundary = float(np.quantile(xs_array, 0.08))
            root_mask = xs_array <= boundary + 1.0
        root_y = float(np.median(ys_array[root_mask])) if root_mask.any() else float(np.median(ys_array))
        return {
            "area": float(len(xs_all)),
            "outer_span": float(median(spans)) if spans else 0.0,
            "root_x": round(boundary, 5),
            "root_y": round(root_y, 5),
            "box": [int(xs_array.min()), int(ys_array.min()), int(xs_array.max()) + 1, int(ys_array.max()) + 1],
        }

    left_side, right_side = side("left"), side("right")
    dominant_name = "left" if left_side["area"] >= right_side["area"] else "right"
    dominant = left_side if dominant_name == "left" else right_side
    all_spans: list[int] = []
    for y in range(y0, y1):
        xs = np.where(mask[y])[0]
        if len(xs):
            all_spans.append(int(xs.max()) - int(xs.min()) + 1)
    return {
        "side": dominant_name if dominant["area"] > 0 else None,
        "area": dominant["area"],
        "outer_span": float(median(all_spans)) if all_spans else 0.0,
        "root_x": dominant["root_x"],
        "root_y": dominant["root_y"],
        "left": left_side,
        "right": right_side,
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-9:
        return None
    return round(float(numerator) / float(denominator), 6)


def symmetry_ratio(left: float | None, right: float | None) -> float | None:
    """Return an order-independent left/right size ratio."""

    if left is None or right is None or min(left, right) <= 0:
        return None
    return round(max(float(left), float(right)) / min(float(left), float(right)), 6)


def metrics(cell: Image.Image) -> dict[str, Any]:
    mask = alpha_mask(cell)
    box = alpha_box(mask)
    if box is None:
        return {"blank": True}
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    alpha_ys, alpha_xs = np.where(mask)
    component_count, components = connected_component_count(mask)
    regions = {name: region_metrics(mask, box, low, high) for name, (low, high) in REGIONS.items()}
    head_width = regions["head"]["outer_span"]
    body_center_x = float(regions["head"]["center_x"])
    ears = ear_silhouette_metrics(mask, box, body_center_x)
    legs = support_metrics(
        mask,
        box,
        head_width,
        body_center_x,
        top_offset=62,
        bottom_offset=16,
        minimum_samples=8,
    )
    paws = support_metrics(
        mask,
        box,
        head_width,
        body_center_x,
        top_offset=20,
        bottom_offset=0,
        allow_center_split=True,
        window_scale=0.58,
        target_offset_ratio=0.18,
        target_clip_ratio=0.22,
        minimum_samples=4,
    )
    forelimb_proxy = forelimb_column_metrics(mask, box, head_width, body_center_x)
    tail = tail_metrics(mask, box, head_width)
    result: dict[str, Any] = {
        "blank": False,
        "box": [left, top, right, bottom],
        "width": width,
        "height": height,
        "aspect": round(width / max(1.0, height), 6),
        "area": int(mask.sum()),
        "center_x": round((left + right - 1) / 2.0, 5),
        "center_y": round((top + bottom - 1) / 2.0, 5),
        "alpha_centroid_x": round(float(alpha_xs.mean()), 5),
        "alpha_centroid_y": round(float(alpha_ys.mean()), 5),
        "baseline": bottom,
        "edge_touch": bool(mask[:, 0].any() or mask[:, -1].any() or mask[0, :].any() or mask[-1, :].any()),
        "component_count": component_count,
        "components": components[:4],
        **regions,
        "ears": ears,
        "legs": legs,
        "forelimb_proxy": forelimb_proxy,
        "paws": paws,
        "tail": tail,
    }
    result["ratios"] = {
        "head_to_body_area": ratio(result["head"]["area"], result["area"]),
        "shoulder_to_head": ratio(result["shoulder"]["outer_span"], result["head"]["outer_span"]),
        "chest_to_head": ratio(result["chest"]["outer_span"], result["head"]["outer_span"]),
        "belly_to_head": ratio(result["belly"]["outer_span"], result["head"]["outer_span"]),
        "lower_to_head": ratio(result["lower"]["outer_span"], result["head"]["outer_span"]),
        "left_leg_to_head": ratio(legs["left"]["width"], result["head"]["outer_span"]),
        "right_leg_to_head": ratio(legs["right"]["width"], result["head"]["outer_span"]),
        "forelimb_left_to_head": ratio(forelimb_proxy["left"]["width"], result["head"]["outer_span"]),
        "forelimb_right_to_head": ratio(forelimb_proxy["right"]["width"], result["head"]["outer_span"]),
        "left_paw_to_head": ratio(paws["left"]["width"], result["head"]["outer_span"]),
        "right_paw_to_head": ratio(paws["right"]["width"], result["head"]["outer_span"]),
        "paw_spacing_to_head": ratio(paws["spacing"], result["head"]["outer_span"]),
    }
    result["symmetry"] = {
        "paws": {
            "ratio": symmetry_ratio(paws["left"]["width"], paws["right"]["width"]),
            "left_width": paws["left"]["width"],
            "right_width": paws["right"]["width"],
            "reliable": bool(paws["reliable"]),
            "limit": SYMMETRY_LIMITS["paws"],
        },
        "legs": {
            "ratio": symmetry_ratio(legs["left"]["width"], legs["right"]["width"]),
            "left_width": legs["left"]["width"],
            "right_width": legs["right"]["width"],
            "reliable": bool(legs["reliable"]),
            "limit": SYMMETRY_LIMITS["legs"],
        },
        "forelimb_proxy": {
            "ratio": symmetry_ratio(forelimb_proxy["left"]["width"], forelimb_proxy["right"]["width"]),
            "left_width": forelimb_proxy["left"]["width"],
            "right_width": forelimb_proxy["right"]["width"],
            "reliable": (
                forelimb_proxy["left"]["width"] is not None
                and forelimb_proxy["right"]["width"] is not None
            ),
            "limit": SYMMETRY_LIMITS["forelimb_proxy"],
        },
    }
    # Compatibility aliases from the original v1 report.
    result["face_width"] = result["face"]["central_span"]
    result["face_area_proxy"] = result["face"]["area"]
    result["shoulder_width"] = result["shoulder"]["central_span"]
    result["torso_width"] = result["chest"]["central_span"]
    result["lower_width"] = result["lower"]["central_span"]
    result["outer_lower_width"] = result["lower"]["outer_span"]
    return result


def scalar_at(document: dict[str, Any], path: str) -> float | None:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def tolerance_for(path: str, target: float) -> float:
    """A phase-target tolerance in pixels or a relative scalar unit."""

    if path.endswith((
        "root_x",
        "root_y",
        "center_x",
        "center_y",
        ".top",
        ".bottom",
        ".spacing",
        "tip_x",
        "tip_y",
        "outer_x",
    )) or path == "center_x":
        return 3.0 if not path.startswith("tail.") else 4.0
    if path == "aspect":
        return 0.045
    if path.startswith("ratios."):
        return max(0.06, abs(target) * 0.12)
    if path.endswith(".area") or path == "area":
        return max(18.0, abs(target) * (0.18 if path.startswith(("legs.", "paws.", "tail.")) else 0.10))
    if path.startswith(("legs.", "paws.", "forelimb_proxy.")):
        return max(3.0, abs(target) * 0.12)
    if path.startswith("tail."):
        return max(4.0, abs(target) * 0.15)
    if path in {"width", "height"}:
        return max(3.0, abs(target) * 0.035)
    if path.startswith("head."):
        return max(3.0, abs(target) * (0.10 if path.endswith("area") else 0.06))
    return max(3.0, abs(target) * 0.08)


def interpolate_metric(first: dict[str, Any], second: dict[str, Any], t: float, path: str) -> float | None:
    a, b = scalar_at(first, path), scalar_at(second, path)
    if a is None or b is None:
        return None
    return a + (b - a) * t


def support_is_reliable(document: dict[str, Any], group: str) -> bool:
    """Only compare a pair when its scanline tracker has real paired support."""

    return bool(document[group].get("reliable", False))


def target_path_is_reliable(first: dict[str, Any], second: dict[str, Any], path: str) -> bool:
    if path.startswith("legs.") or path.startswith("ratios.left_leg") or path.startswith("ratios.right_leg"):
        return support_is_reliable(first, "legs") and support_is_reliable(second, "legs")
    if path.startswith("paws.") or path.startswith("ratios.left_paw") or path.startswith("ratios.right_paw") or path.startswith("ratios.paw_spacing"):
        return support_is_reliable(first, "paws") and support_is_reliable(second, "paws")
    return True


def parse_mode_map(path: Path, expected_frames: list[str]) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("frames", raw)
    raw_specs: dict[str, Any]
    if isinstance(items, list):
        raw_specs = {str(item["frame"]): item for item in items}
    elif isinstance(items, dict):
        raw_specs = {str(key): value for key, value in items.items()}
    else:
        raise ValueError("mode map must contain a frames object or list")
    missing = [frame for frame in expected_frames if frame not in raw_specs]
    extra = sorted(set(raw_specs) - set(expected_frames))
    if missing or extra:
        raise ValueError(f"invalid mode map: missing={missing}; extra={extra}")
    result: dict[str, dict[str, Any]] = {}
    for frame in expected_frames:
        raw_spec = raw_specs[frame]
        spec = {"mode": raw_spec} if isinstance(raw_spec, str) else dict(raw_spec) if isinstance(raw_spec, dict) else None
        if spec is None:
            raise ValueError(f"invalid mode specification for {frame}")
        mode = str(spec.get("mode", ""))
        pair = spec.get("anchor_pair")
        if pair is not None:
            pair = str(pair)
        progress = spec.get("t", spec.get("progress"))
        if mode not in MODES:
            raise ValueError(f"invalid mode for {frame}: {mode!r}")
        if pair is not None and pair not in ANCHOR_PAIRS:
            raise ValueError(f"invalid anchor_pair for {frame}: {pair!r}")
        if pair is not None:
            if mode != "transition":
                raise ValueError(f"anchor_pair is valid only for transition frames: {frame}")
            try:
                progress = float(progress)
            except (TypeError, ValueError) as error:
                raise ValueError(f"transition {frame} requires numeric t in [0,1]") from error
            if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
                raise ValueError(f"transition {frame} has invalid t={progress!r}")
        elif progress is not None:
            raise ValueError(f"transition progress requires anchor_pair: {frame}")
        result[frame] = {"mode": mode, "anchor_pair": pair, "t": progress}
    last_progress: dict[str, tuple[str, float]] = {}
    for frame in expected_frames:
        pair, progress = result[frame]["anchor_pair"], result[frame]["t"]
        if pair is None:
            continue
        assert progress is not None
        if pair in last_progress:
            previous_frame, previous_progress = last_progress[pair]
            if progress + 1e-12 < previous_progress:
                raise ValueError(
                    f"anchor progress must be nondecreasing for {pair}: "
                    f"{previous_frame} t={previous_progress:.6f} -> {frame} t={progress:.6f}"
                )
        last_progress[pair] = (frame, progress)
    return result


def compare_to_target(
    name: str,
    actual: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    t: float,
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    for path in TARGET_PATHS:
        if not target_path_is_reliable(first, second, path):
            continue
        expected = interpolate_metric(first, second, t, path)
        if expected is None:
            continue
        observed = scalar_at(actual, path)
        if observed is None:
            failures.append(f"{name}:{label}:{path}=missing")
            continue
        tolerance = tolerance_for(path, expected)
        if abs(observed - expected) > tolerance:
            failures.append(
                f"{name}:{label}:{path}={observed:.3f}; expected={expected:.3f}±{tolerance:.3f}"
            )
    return failures


def symmetry_failures(name: str, value: dict[str, Any]) -> list[str]:
    """Reject visibly unequal supports in front-facing interaction poses.

    Run-based paw and leg measurements are checked only when their paired
    tracker is reliable, so a genuinely occluded or merged support does not
    become a false hard failure.  The fixed forelimb proxy is checked whenever
    both anatomical columns contain measurable pixels.
    """

    failures: list[str] = []
    for group, limit in SYMMETRY_LIMITS.items():
        measurement = value["symmetry"][group]
        observed = measurement["ratio"]
        if not measurement["reliable"] or observed is None:
            continue
        if observed > limit:
            failures.append(
                f"{name}:symmetry_{group}_ratio={observed:.3f}; limit={limit:.3f}; "
                f"left={measurement['left_width']:.3f}; right={measurement['right_width']:.3f}"
            )
    return failures


def generic_identity_failures(name: str, value: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    """Strict invariant checks that also protect old, phase-less mode maps."""

    failures: list[str] = []
    failures.extend(symmetry_failures(name, value))
    if value["component_count"] != 1:
        failures.append(f"{name}:components={value['component_count']}")
    if value["edge_touch"]:
        failures.append(f"{name}:touches_cell_edge")
    if value["baseline"] != BASELINE:
        failures.append(f"{name}:baseline={value['baseline']}; expected={BASELINE}")
    for side in ("left", "right"):
        bottom = value["paws"][side]["bottom"]
        # Seated poses can merge paws into the body and do not always expose a
        # pair.  When a paw is detectable, however, it must share the floor.
        if bottom is not None and bottom < BASELINE - 1:
            failures.append(f"{name}:paw_{side}_bottom={bottom}; expected≥{BASELINE - 1}")
    for path, relative in (("face.outer_span", 0.10), ("head.area", 0.16), ("head.height", 0.10)):
        observed, reference = scalar_at(value, path), scalar_at(canonical, path)
        if observed is None or reference is None:
            continue
        if abs(observed - reference) > max(3.0, reference * relative):
            failures.append(f"{name}:identity_{path}={observed:.3f}; canonical={reference:.3f}")
    shoulder_ratio = scalar_at(value, "ratios.shoulder_to_head")
    if shoulder_ratio is not None and not 1.03 <= shoulder_ratio <= 1.38:
        failures.append(f"{name}:shoulder_head_ratio={shoulder_ratio:.3f}")
    return failures


def failures_for_frame(
    name: str,
    spec: dict[str, Any],
    value: dict[str, Any],
    canonical: dict[str, Any],
    half_rise: dict[str, Any] | None,
    standing: dict[str, Any] | None,
) -> list[str]:
    if value.get("blank"):
        return [f"{name}:blank"]
    failures = generic_identity_failures(name, value, canonical)
    mode, pair, t = spec["mode"], spec["anchor_pair"], spec["t"]
    if mode == "seated":
        failures.extend(compare_to_target(name, value, canonical, canonical, 0.0, label="C"))
    elif mode == "compact-standing":
        if standing is None:
            failures.append(f"{name}:missing_standing_master")
        else:
            failures.extend(compare_to_target(name, value, standing, standing, 0.0, label="S"))
    elif pair == "C_H":
        if half_rise is None:
            failures.append(f"{name}:missing_half_rise_master")
        else:
            failures.extend(compare_to_target(name, value, canonical, half_rise, float(t), label=f"C_H@{t:.3f}"))
    elif pair == "H_S":
        if half_rise is None:
            failures.append(f"{name}:missing_half_rise_master")
        elif standing is None:
            failures.append(f"{name}:missing_standing_master")
        else:
            failures.extend(compare_to_target(name, value, half_rise, standing, float(t), label=f"H_S@{t:.3f}"))
    return failures


def temporal_limit(path: str, *, cross_phase: bool) -> tuple[str, float]:
    """Return (kind, threshold): relative changes or absolute pixel movement."""

    if path.endswith((
        "root_x",
        "root_y",
        "center_x",
        "center_y",
        ".top",
        ".bottom",
        ".spacing",
        "tip_x",
        "tip_y",
        "outer_x",
    )) or path == "center_x":
        return "absolute", 4.0 if cross_phase else 3.0
    if path == "aspect":
        return "relative", 0.08 if cross_phase else 0.06
    if path.endswith(".area") or path == "area":
        return "relative", 0.18 if path.startswith(("legs.", "paws.", "tail.")) else (0.10 if cross_phase else 0.08)
    if path.startswith(("legs.", "paws.", "forelimb_proxy.")):
        return "relative", 0.16 if cross_phase else 0.12
    if path.startswith("tail."):
        return "relative", 0.18 if cross_phase else 0.14
    if path.startswith("head."):
        return "relative", 0.08 if cross_phase else 0.06
    return "relative", 0.08 if cross_phase else 0.06


def delta(value_a: float, value_b: float, kind: str) -> float:
    if kind == "absolute":
        return abs(value_b - value_a)
    return abs(value_b - value_a) / max(1.0, abs(value_a))


def temporal_failures(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for previous, current in zip(frames, frames[1:]):
        a, b = previous["metrics"], current["metrics"]
        cross_phase = (
            previous["mode"] != current["mode"]
            or previous["anchor_pair"] != current["anchor_pair"]
        )
        failures: list[str] = []
        changes: dict[str, float] = {}
        for path in TEMPORAL_PATHS:
            first, second = scalar_at(a, path), scalar_at(b, path)
            if first is None or second is None:
                continue
            kind, limit = temporal_limit(path, cross_phase=cross_phase)
            amount = delta(first, second, kind)
            changes[path] = round(amount, 6)
            if amount > limit:
                failures.append(f"{path}_change={amount:.3f}; limit={limit:.3f}")
        records.append({
            "from": previous["frame"],
            "to": current["frame"],
            "cross_phase": cross_phase,
            "changes": changes,
            "failures": failures,
        })
    return records


def three_frame_trends(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject a one-frame bulge/shrink that may evade pairwise limits."""

    records: list[dict[str, Any]] = []
    for before, current, after in zip(frames, frames[1:], frames[2:]):
        context = (before["mode"], before["anchor_pair"])
        same_context = (
            context == (current["mode"], current["anchor_pair"])
            and context == (after["mode"], after["anchor_pair"])
        )
        failures: list[str] = []
        residuals: dict[str, float] = {}
        if same_context:
            for path in TEMPORAL_PATHS:
                a = scalar_at(before["metrics"], path)
                b = scalar_at(current["metrics"], path)
                c = scalar_at(after["metrics"], path)
                if a is None or b is None or c is None:
                    continue
                kind, _limit = temporal_limit(path, cross_phase=False)
                midpoint = (a + c) / 2.0
                amount = abs(b - midpoint) if kind == "absolute" else abs(b - midpoint) / max(1.0, abs(midpoint))
                residuals[path] = round(amount, 6)
                limit = 4.0 if kind == "absolute" else (0.16 if path.startswith(("legs.", "paws.", "tail.")) else 0.10)
                if amount > limit:
                    failures.append(f"{path}_local_residual={amount:.3f}; limit={limit:.3f}")

        reversals: dict[str, dict[str, float]] = {}
        for path in REVERSAL_PATHS:
            a = scalar_at(before["metrics"], path)
            b = scalar_at(current["metrics"], path)
            c = scalar_at(after["metrics"], path)
            if a is None or b is None or c is None:
                continue
            first_delta, second_delta = b - a, c - b
            if first_delta * second_delta >= 0.0:
                continue
            backtrack = min(abs(first_delta), abs(second_delta))
            rolling_range = max(a, b, c) - min(a, b, c)
            noise = reversal_noise(path, (a + b + c) / 3.0)
            reversals[path] = {
                "first_delta": round(first_delta, 6),
                "second_delta": round(second_delta, 6),
                "backtrack": round(backtrack, 6),
                "rolling_range": round(rolling_range, 6),
                "noise": round(noise, 6),
            }
            if backtrack > noise:
                failures.append(
                    f"{path}_direction_reversal={backtrack:.3f}; "
                    f"rolling_range={rolling_range:.3f}; noise={noise:.3f}"
                )
        records.append({
            "before": before["frame"],
            "frame": current["frame"],
            "after": after["frame"],
            "context": {"mode": context[0], "anchor_pair": context[1]},
            "same_context": same_context,
            "residuals": residuals,
            "reversals": reversals,
            "failures": failures,
        })
    return records


def reversal_noise(path: str, midpoint: float) -> float:
    """Ignore one-pixel matte quantisation, but catch a visible rebound."""

    if path.startswith("ratios."):
        return 0.02
    if path in {"alpha_centroid_x", "alpha_centroid_y"} or path.endswith("center_x"):
        return 0.75
    if path.startswith("paws."):
        return max(1.5, abs(midpoint) * 0.04)
    if path.startswith("forelimb_proxy."):
        return max(1.5, abs(midpoint) * 0.04)
    return max(1.5, abs(midpoint) * 0.025)


def anchor_progress_noise(path: str, endpoint: float) -> float:
    """Tolerate tiny anatomical motion without allowing a visible reversal."""

    if path.endswith(".area") or path == "area":
        relative = 0.04 if path.startswith(("paws.", "tail.")) else 0.025
        return max(18.0, abs(endpoint) * relative)
    if path.startswith(("paws.", "tail.")):
        return max(1.0, abs(endpoint) * 0.035)
    if path in {"width", "height"}:
        return max(1.0, abs(endpoint) * 0.015)
    return max(1.0, abs(endpoint) * 0.035)


def anchor_progression_failures(
    frames: list[dict[str, Any]],
    half_rise: dict[str, Any] | None,
    standing: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Reject a material step away from an advancing phase's end anchor."""

    endpoints = {"C_H": half_rise, "H_S": standing}
    records: list[dict[str, Any]] = []
    away_runs: dict[tuple[str, str], dict[str, Any]] = {}
    for previous, current in zip(frames, frames[1:]):
        pair = previous["anchor_pair"]
        if pair is None or pair != current["anchor_pair"]:
            away_runs.clear()
            continue
        previous_t, current_t = previous["t"], current["t"]
        assert previous_t is not None and current_t is not None
        if current_t <= previous_t + 1e-12:
            continue
        endpoint = endpoints[pair]
        if endpoint is None:
            continue
        failures: list[str] = []
        distances: dict[str, dict[str, float]] = {}
        for path in ANCHOR_PROGRESS_PATHS:
            if not target_path_is_reliable(endpoint, endpoint, path):
                continue
            target = scalar_at(endpoint, path)
            before = scalar_at(previous["metrics"], path)
            after = scalar_at(current["metrics"], path)
            if target is None or before is None or after is None:
                continue
            before_distance = abs(before - target)
            after_distance = abs(after - target)
            regression = after_distance - before_distance
            noise = anchor_progress_noise(path, target)
            distances[path] = {
                "before": round(before_distance, 6),
                "after": round(after_distance, 6),
                "regression": round(regression, 6),
                "noise": round(noise, 6),
            }
            if regression > noise:
                failures.append(
                    f"{path}_moves_away={regression:.3f}; noise={noise:.3f}; endpoint={target:.3f}"
                )
            key = (pair, path)
            if regression > 0:
                state = away_runs.setdefault(
                    key,
                    {"count": 0, "start_distance": before_distance, "reported": False},
                )
                state["count"] += 1
                cumulative = after_distance - float(state["start_distance"])
                if state["count"] >= 2 and cumulative > noise and not state["reported"]:
                    failures.append(
                        f"{path}_persistently_moves_away={cumulative:.3f}; "
                        f"steps={state['count']}; noise={noise:.3f}; endpoint={target:.3f}"
                    )
                    state["reported"] = True
            else:
                away_runs.pop(key, None)
        records.append(
            {
                "from": previous["frame"],
                "to": current["frame"],
                "anchor_pair": pair,
                "t": [previous_t, current_t],
                "distances": distances,
                "failures": failures,
            }
        )
    return records


def read_cell(path: Path, *, label: str) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.size != (CELL_W, CELL_H):
        raise SystemExit(f"{label}_size={image.size}; expected={(CELL_W, CELL_H)}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate interaction anatomy, C/H/S phase targets, and temporal continuity.")
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path, help="Approved C 192x208 neutral crouch cell.")
    parser.add_argument("--half-rise-master", type=Path, help="Approved H 192x208 half-rise cell used by C_H/H_S phase targets.")
    parser.add_argument("--standing-master", type=Path, help="Approved S 192x208 compact-standing cell.")
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--columns", required=True, type=int)
    parser.add_argument("--mode-map", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0:
        raise SystemExit("--rows and --columns must be positive")
    atlas = Image.open(args.atlas).convert("RGBA")
    expected_size = (CELL_W * args.columns, CELL_H * args.rows)
    if atlas.size != expected_size:
        raise SystemExit(f"atlas_size={atlas.size}; expected={expected_size}")
    canonical_metrics = metrics(read_cell(args.canonical, label="canonical"))
    half_metrics = metrics(read_cell(args.half_rise_master, label="half_rise_master")) if args.half_rise_master else None
    standing_metrics = metrics(read_cell(args.standing_master, label="standing_master")) if args.standing_master else None
    endpoint_failures = {
        "canonical": symmetry_failures("canonical", canonical_metrics),
        "half_rise_master": [] if half_metrics is None else symmetry_failures("half_rise_master", half_metrics),
        "standing_master": [] if standing_metrics is None else symmetry_failures("standing_master", standing_metrics),
    }
    names = [f"F{index:02d}" for index in range(args.rows * args.columns)]
    mode_map = parse_mode_map(args.mode_map, names)
    frame_records: list[dict[str, Any]] = []
    failures: list[str] = [failure for group in endpoint_failures.values() for failure in group]
    for index, name in enumerate(names):
        row, column = divmod(index, args.columns)
        cell = atlas.crop((column * CELL_W, row * CELL_H, (column + 1) * CELL_W, (row + 1) * CELL_H))
        value = metrics(cell)
        spec = mode_map[name]
        frame_failures = failures_for_frame(name, spec, value, canonical_metrics, half_metrics, standing_metrics)
        failures.extend(frame_failures)
        frame_records.append({"frame": name, **spec, "metrics": value, "failures": frame_failures})
    temporal = temporal_failures(frame_records)
    trends = three_frame_trends(frame_records)
    anchor_progression = anchor_progression_failures(frame_records, half_metrics, standing_metrics)
    for record in temporal:
        failures.extend(f"{record['from']}->{record['to']}:{failure}" for failure in record["failures"])
    for record in trends:
        failures.extend(f"{record['before']}->{record['frame']}->{record['after']}:{failure}" for failure in record["failures"])
    for record in anchor_progression:
        failures.extend(f"{record['from']}->{record['to']}:{failure}" for failure in record["failures"])
    report = {
        "schema": "desktop-pet.interaction-proportions.v2",
        "ok": not failures,
        "atlas": str(args.atlas),
        "canonical": str(args.canonical),
        "half_rise_master": None if args.half_rise_master is None else str(args.half_rise_master),
        "standing_master": None if args.standing_master is None else str(args.standing_master),
        "rows": args.rows,
        "columns": args.columns,
        "canonical_metrics": canonical_metrics,
        "half_rise_metrics": half_metrics,
        "standing_metrics": standing_metrics,
        "endpoint_failures": endpoint_failures,
        "frames": frame_records,
        "temporal": temporal,
        "three_frame_trends": trends,
        "anchor_progression": anchor_progression,
        "failures": failures,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
