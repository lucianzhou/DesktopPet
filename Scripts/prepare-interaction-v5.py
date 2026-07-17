#!/usr/bin/env python3
"""Prepare one coherent 64-frame or 48-frame Baomihua interaction storyboard.

This is deliberately a *registration* pipeline, not an image-repair pipeline.
It accepts only one flat-magenta storyboard containing either 8x8/64 or
8x6/48 fully separated cats, selected explicitly by ``--rows`` and
``--columns``. Every generated pose keeps its complete silhouette; the only
geometric operation is a single shared scale for the whole storyboard, a
preserved horizontal source offset, and one shared foot baseline. The approved
runtime neutral is copied byte-for-byte only into the first and final frames.

There is intentionally no per-frame fit, no body locking, no face overlay, no
cross-frame compositing, and no fallback that turns an invalid storyboard into
an atlas.  Run ``validate-interaction-v5.py`` after this script before using an
atlas in the application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # scipy makes connected-component recovery much faster on large boards.
    from scipy import ndimage as scipy_ndimage
except ImportError:  # The pure NumPy/Python fallback keeps the script portable.
    scipy_ndimage = None


CELL_WIDTH = 192
CELL_HEIGHT = 208
DEFAULT_COLUMNS = 8
DEFAULT_ROWS = 8
SUPPORTED_GRIDS = {(8, 8), (6, 8)}  # (rows, columns): 64 or 48 frames only.

# Matches the registered desktop-pet geometry already used by the app.  A full
# standing pose may be 196px tall, which still leaves 8px at the top and 4px
# below the shared 204px foot baseline.
CONTENT_WIDTH = 176
CONTENT_HEIGHT = 196
LEFT_PADDING = (CELL_WIDTH - CONTENT_WIDTH) // 2
RIGHT_PADDING = CELL_WIDTH - LEFT_PADDING - CONTENT_WIDTH
TOP_PADDING = CELL_HEIGHT - 4 - CONTENT_HEIGHT
BOTTOM_PADDING = 4
BASELINE_Y = CELL_HEIGHT - BOTTOM_PADDING
VISIBLE_ALPHA = 8
# LANCZOS is useful for premultiplied colour detail, but its negative lobes can
# create a one-pixel alpha ring several pixels away from a silhouette.  Alpha
# uses a non-ringing filter instead, then values at or below the component
# visibility threshold are made genuinely transparent.
ALPHA_RESAMPLER = Image.Resampling.BILINEAR
MAX_TINY_RESAMPLING_ISLAND_PIXELS = 6

# Keep these thresholds aligned with prepare-canonical.py.  Image generation
# often emits a clean-looking magenta *brightness gradient* rather than exact
# #FF00FF, so chroma classification must be driven by colour separation, not
# absolute red/blue channels.  A real pink nose or cream fur has a much lower
# ``min(red, blue) - green`` score and remains opaque.
KEY_HARD_SCORE = 75
KEY_SOFT_SCORE = 35
KEY_EDGE_DESPILL_SCORE = 25
MAX_CORNER_RGB_VARIATION = 48.0
MAX_BORDER_PLANE_RESIDUAL_P99 = 16.0
MAX_BORDER_PLANE_RESIDUAL_HIGH_RATIO = 0.01
# The first and final storyboard slots depict the same canonical front-seated
# pose.  Comparing them catches a slow camera/scale drift that can evade every
# adjacent-frame continuity threshold while still looking obviously smaller by
# the end of the five-second interaction.
MAX_NEUTRAL_ENDPOINT_DIMENSION_RATIO = 1.08
MAX_NEUTRAL_ENDPOINT_AREA_RATIO = 1.12


class StoryboardContractError(ValueError):
    """An actionable rejection of a storyboard that cannot be safely split."""

    def __init__(self, reasons: Iterable[str], diagnostics: dict[str, Any] | None = None):
        self.reasons = list(reasons)
        self.diagnostics = diagnostics or {}
        super().__init__("; ".join(self.reasons))


@dataclass
class Component:
    """One foreground component recovered from the keyed storyboard."""

    label: int
    box: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]
    slot_row: int = -1
    slot_column: int = -1

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass
class StoryboardAnalysis:
    path: Path
    raw_size: tuple[int, int]
    keyed: Image.Image
    labels: np.ndarray
    components: list[Component]  # Strict row-major F00 ... F(last) order.
    flat_background: dict[str, Any]
    slot_size: tuple[float, float]
    rows: int
    columns: int

    @property
    def frame_count(self) -> int:
        return self.rows * self.columns

    @property
    def atlas_size(self) -> tuple[int, int]:
        return (CELL_WIDTH * self.columns, CELL_HEIGHT * self.rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value: Any) -> Any:
    """Convert NumPy scalar values before serialising a QA report."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def require_supported_grid(rows: int, columns: int) -> None:
    """Refuse to guess a layout from component count or source aspect ratio."""

    if (rows, columns) not in SUPPORTED_GRIDS:
        supported = ", ".join(f"{columns_value}x{rows_value}" for rows_value, columns_value in sorted(SUPPORTED_GRIDS))
        raise ValueError(
            f"unsupported_grid={columns}x{rows}; explicitly pass one supported grid: {supported}. "
            "The pipeline never infers rows/columns from a component count."
        )


def clear_transparent_rgb(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    array[array[..., 3] == 0, :3] = 0
    return Image.fromarray(array, "RGBA")


def magenta_score(rgb: np.ndarray) -> np.ndarray:
    """How strongly an RGB pixel resembles the required magenta key."""

    signed = rgb.astype(np.int16)
    return np.minimum(signed[..., 0], signed[..., 2]) - signed[..., 1]


def key_candidate_mask(array: np.ndarray, *, minimum_score: int = KEY_SOFT_SCORE) -> np.ndarray:
    """Score-based chroma classifier shared by source inspection and keying."""

    alpha = array[..., 3]
    score = magenta_score(array[..., :3])
    return (alpha >= 200) & (score >= minimum_score)


def visible_key_spill_count(image: Image.Image) -> int:
    """Count remaining strong key-colour pixels, not ordinary pink fur."""

    array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    score = magenta_score(array[..., :3])
    spill = (array[..., 3] > VISIBLE_ALPHA) & (score >= KEY_HARD_SCORE)
    return int(np.count_nonzero(spill))


def flat_magenta_diagnostics(raw: Image.Image) -> dict[str, Any]:
    """Validate a smooth, strongly-magenta generated backing before keying.

    The background may brighten or darken smoothly across the canvas.  It may
    not be a checkerboard, texture, scenery, or a different colour field.
    """

    array = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    height, width = array.shape[:2]
    block = max(1, min(8, height // 16, width // 16))
    corner_slices = {
        "top_left": (slice(0, block), slice(0, block)),
        "top_right": (slice(0, block), slice(width - block, width)),
        "bottom_left": (slice(height - block, height), slice(0, block)),
        "bottom_right": (slice(height - block, height), slice(width - block, width)),
    }
    key_mask = key_candidate_mask(array, minimum_score=KEY_HARD_SCORE)
    corners: dict[str, Any] = {}
    corner_means: list[np.ndarray] = []
    for name, (ys, xs) in corner_slices.items():
        pixels = array[ys, xs]
        candidate_ratio = float(np.mean(key_mask[ys, xs]))
        mean_rgb = pixels[..., :3].reshape(-1, 3).mean(axis=0)
        corner_means.append(mean_rgb)
        corners[name] = {
            "key_ratio": round(candidate_ratio, 6),
            "mean_rgb": [round(float(channel), 3) for channel in mean_rgb],
        }
    means = np.stack(corner_means)
    variation = float(np.max(means.max(axis=0) - means.min(axis=0)))
    border_thickness = max(2, min(16, min(height, width) // 64))
    border = np.zeros((height, width), dtype=bool)
    border[:border_thickness, :] = True
    border[-border_thickness:, :] = True
    border[:, :border_thickness] = True
    border[:, -border_thickness:] = True
    border_key = border & key_mask
    border_key_ratio = float(np.mean(key_mask[border]))
    ys, xs = np.where(border_key)
    plane_residual_p99: float | None = None
    plane_residual_high_ratio: float | None = None
    if len(xs) >= 16:
        # Cap the regression input deterministically for unusually large source
        # boards while retaining samples from every side of the border.
        step = max(1, len(xs) // 100_000)
        ys = ys[::step]
        xs = xs[::step]
        design = np.column_stack((np.ones(len(xs)), xs / max(1, width - 1), ys / max(1, height - 1)))
        rgb_samples = array[ys, xs, :3].astype(np.float64)
        coefficients, _, _, _ = np.linalg.lstsq(design, rgb_samples, rcond=None)
        residual = np.max(np.abs(rgb_samples - design @ coefficients), axis=1)
        plane_residual_p99 = float(np.percentile(residual, 99))
        plane_residual_high_ratio = float(np.mean(residual > 24.0))
    return {
        "corner_block_px": block,
        "corner_samples": corners,
        "corner_rgb_max_variation": round(variation, 5),
        "hard_magenta_pixel_ratio": round(float(np.mean(key_mask)), 6),
        "border_thickness_px": border_thickness,
        "border_hard_magenta_ratio": round(border_key_ratio, 6),
        "border_linear_gradient_residual_p99": None if plane_residual_p99 is None else round(plane_residual_p99, 5),
        "border_linear_gradient_high_residual_ratio": None
        if plane_residual_high_ratio is None
        else round(plane_residual_high_ratio, 6),
    }


def require_flat_magenta_background(raw: Image.Image) -> dict[str, Any]:
    diagnostics = flat_magenta_diagnostics(raw)
    reasons: list[str] = []
    weak_corners = [
        name
        for name, item in diagnostics["corner_samples"].items()
        if item["key_ratio"] < 0.98
    ]
    if weak_corners:
        reasons.append(
            "storyboard_background_not_flat_magenta_at_corners:"
            + ",".join(weak_corners)
            + " (use an opaque, smooth strong-magenta generated background)"
        )
    if diagnostics["corner_rgb_max_variation"] > MAX_CORNER_RGB_VARIATION:
        reasons.append(
            "storyboard_background_corner_brightness_variation_too_large:"
            f"variation={diagnostics['corner_rgb_max_variation']} (remove textures/checkerboards/scenery)"
        )
    if diagnostics["border_hard_magenta_ratio"] < 0.95:
        reasons.append(
            "storyboard_border_is_not_predominantly_strong_magenta:"
            f"ratio={diagnostics['border_hard_magenta_ratio']} (use a full generated-magenta backing)"
        )
    residual_p99 = diagnostics["border_linear_gradient_residual_p99"]
    high_residual_ratio = diagnostics["border_linear_gradient_high_residual_ratio"]
    if residual_p99 is None or high_residual_ratio is None:
        reasons.append("storyboard_border_has_too_few_magenta_samples_for_smoothness_check")
    elif residual_p99 > MAX_BORDER_PLANE_RESIDUAL_P99 or high_residual_ratio > MAX_BORDER_PLANE_RESIDUAL_HIGH_RATIO:
        reasons.append(
            "storyboard_background_is_textured_or_checkerboard_like:"
            f"p99_residual={residual_p99},high_residual_ratio={high_residual_ratio} "
            "(keep only a smooth magenta brightness gradient)"
        )
    if diagnostics["hard_magenta_pixel_ratio"] < 0.02:
        reasons.append(
            "storyboard_has_too_little_smooth_magenta_background "
            "(source must be one explicit 8x8 or 8x6 board, not a transparent atlas or cropped strip)"
        )
    if reasons:
        raise StoryboardContractError(reasons, {"flat_background": diagnostics})
    return diagnostics


def _despill_key_edges(array: np.ndarray) -> np.ndarray:
    """Replace semitransparent key RGB with nearby opaque sprite RGB.

    This is a chroma-key cleanup only.  It never adds, moves, crops, or blends
    a body part from another frame.  The alpha matte remains unchanged.
    """

    output = array.copy()
    alpha = output[..., 3]
    colors = output[..., :3].copy()
    score = magenta_score(colors)
    known = alpha >= 250
    edge_band = np.zeros_like(known)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        edge_band |= np.roll(np.roll(alpha < 250, dy, axis=0), dx, axis=1)
    # This mirrors prepare-canonical.py: only pixels on the silhouette
    # boundary whose score leans toward the key are recoloured.  Pink nose and
    # inner-ear details are inside the body silhouette and therefore remain.
    fringe = edge_band & (score > KEY_EDGE_DESPILL_SCORE) & (alpha > 0)
    known &= ~fringe
    pending = ((alpha > 0) & ~known) | fringe
    if not pending.any() or not known.any():
        output[alpha == 0, :3] = 0
        return output

    height, width = alpha.shape
    visited = known.copy()
    queue: deque[tuple[int, int]] = deque((int(y), int(x)) for y, x in np.argwhere(known))
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not pending[ny, nx] or visited[ny, nx]:
                continue
            colors[ny, nx] = colors[y, x]
            visited[ny, nx] = True
            queue.append((ny, nx))
    colors[alpha == 0] = 0
    return np.dstack((colors, alpha))


def remove_magenta_key(raw: Image.Image) -> Image.Image:
    """Turn the flat magenta source background into clean RGBA transparency."""

    array = np.asarray(raw.convert("RGBA"), dtype=np.uint8).copy()
    score = magenta_score(array[..., :3])
    matte = np.ones(score.shape, dtype=np.float32)
    hard = score >= KEY_HARD_SCORE
    soft = (score > KEY_SOFT_SCORE) & ~hard
    matte[hard] = 0.0
    matte[soft] = (KEY_HARD_SCORE - score[soft]) / float(KEY_HARD_SCORE - KEY_SOFT_SCORE)
    array[..., 3] = np.rint(array[..., 3].astype(np.float32) * matte).astype(np.uint8)
    array = _despill_key_edges(array)
    return Image.fromarray(array, "RGBA")


def _component_records_from_labels(labels: np.ndarray, component_count: int) -> list[Component]:
    components: list[Component] = []
    # ``find_objects`` bounds each label first, avoiding 64 full-canvas scans
    # when a 4K storyboard is decoded through scipy.
    objects = scipy_ndimage.find_objects(labels) if scipy_ndimage is not None else [None] * component_count
    for label in range(1, component_count + 1):
        bounds = objects[label - 1]
        if bounds is None:
            ys, xs = np.where(labels == label)
            x_offset = y_offset = 0
        else:
            y_slice, x_slice = bounds
            local = labels[y_slice, x_slice] == label
            ys, xs = np.where(local)
            x_offset = int(x_slice.start)
            y_offset = int(y_slice.start)
        if len(xs) == 0:
            continue
        xs = xs + x_offset
        ys = ys + y_offset
        components.append(
            Component(
                label=label,
                box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                area=int(len(xs)),
                centroid=(float(xs.mean()), float(ys.mean())),
            )
        )
    return components


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[Component]]:
    """Return all 8-connected components, including tiny stray foreground bits."""

    if mask.dtype != bool:
        mask = mask.astype(bool)
    if scipy_ndimage is not None:
        labels, count = scipy_ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        labels = labels.astype(np.int32, copy=False)
        return labels, _component_records_from_labels(labels, int(count))

    # Portable run-length fallback for the bundled Pillow/NumPy runtime when
    # scipy is not available.  A dense cat silhouette then costs one run per
    # scanline instead of one Python queue operation per visible pixel.
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    parent: list[int] = []
    rank: list[int] = []
    runs: list[tuple[int, int, int, int]] = []  # y, start, end-exclusive, run id

    def make_set() -> int:
        identifier = len(parent)
        parent.append(identifier)
        rank.append(0)
        return identifier

    def find(identifier: int) -> int:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    previous: list[tuple[int, int, int]] = []  # start, end-exclusive, run id
    for y in range(height):
        row = mask[y]
        starts = np.flatnonzero(row & np.concatenate(([True], ~row[:-1])))
        ends = np.flatnonzero(row & np.concatenate((~row[1:], [True]))) + 1
        current: list[tuple[int, int, int]] = []
        previous_start = 0
        for start_raw, end_raw in zip(starts, ends):
            start = int(start_raw)
            end = int(end_raw)
            run_id = make_set()
            # Two runs on adjacent rows are 8-connected when their intervals
            # overlap or touch diagonally: prev.end >= start and prev.start <= end.
            while previous_start < len(previous) and previous[previous_start][1] < start:
                previous_start += 1
            candidate = previous_start
            while candidate < len(previous) and previous[candidate][0] <= end:
                union(run_id, previous[candidate][2])
                candidate += 1
            current.append((start, end, run_id))
            runs.append((y, start, end, run_id))
        previous = current

    root_to_label: dict[int, int] = {}
    statistics: dict[int, dict[str, float | int]] = {}
    for y, start, end, run_id in runs:
        root = find(run_id)
        label = root_to_label.setdefault(root, len(root_to_label) + 1)
        labels[y, start:end] = label
        count = end - start
        stats = statistics.setdefault(
            label,
            {
                "area": 0,
                "min_x": start,
                "max_x": end - 1,
                "min_y": y,
                "max_y": y,
                "sum_x": 0.0,
                "sum_y": 0.0,
            },
        )
        stats["area"] = int(stats["area"]) + count
        stats["min_x"] = min(int(stats["min_x"]), start)
        stats["max_x"] = max(int(stats["max_x"]), end - 1)
        stats["min_y"] = min(int(stats["min_y"]), y)
        stats["max_y"] = max(int(stats["max_y"]), y)
        stats["sum_x"] = float(stats["sum_x"]) + (start + end - 1) * count / 2.0
        stats["sum_y"] = float(stats["sum_y"]) + y * count

    components = [
        Component(
            label=label,
            box=(int(stats["min_x"]), int(stats["min_y"]), int(stats["max_x"]) + 1, int(stats["max_y"]) + 1),
            area=int(stats["area"]),
            centroid=(float(stats["sum_x"]) / int(stats["area"]), float(stats["sum_y"]) / int(stats["area"])),
        )
        for label, stats in sorted(statistics.items())
    ]
    return labels, components


def alpha_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def source_slot_bounds(
    row: int,
    column: int,
    source_size: tuple[int, int],
    *,
    rows: int,
    columns: int,
) -> tuple[float, float, float, float]:
    width, height = source_size
    slot_width = width / columns
    slot_height = height / rows
    return (
        column * slot_width,
        row * slot_height,
        (column + 1) * slot_width,
        (row + 1) * slot_height,
    )


def analyze_storyboard(path: Path, *, rows: int = DEFAULT_ROWS, columns: int = DEFAULT_COLUMNS) -> StoryboardAnalysis:
    """Recover exactly one full cat in each row-major storyboard slot."""

    require_supported_grid(rows, columns)
    raw = Image.open(path).convert("RGBA")
    width, height = raw.size
    expected_frames = rows * columns
    if width < columns * 16 or height < rows * 16:
        raise StoryboardContractError(
            [
                f"storyboard_too_small:{width}x{height}; expected one complete {columns}x{rows} board, "
                "not an already-sliced atlas"
            ]
        )
    flat_background = require_flat_magenta_background(raw)
    keyed = remove_magenta_key(raw)
    keyed_array = np.asarray(keyed, dtype=np.uint8)
    mask = keyed_array[..., 3] > VISIBLE_ALPHA
    labels, components = connected_components(mask)
    diagnostics: dict[str, Any] = {
        "source_size": [width, height],
        "foreground_component_count": len(components),
        "flat_background": flat_background,
        "components": [
            {
                "label": component.label,
                "box": list(component.box),
                "area": component.area,
                "centroid": [round(component.centroid[0], 4), round(component.centroid[1], 4)],
            }
            for component in components
        ],
    }
    reasons: list[str] = []
    if len(components) != expected_frames:
        reasons.append(
            f"storyboard_component_count={len(components)}; expected exactly {expected_frames} disconnected full cats "
            f"for explicit {columns}x{rows} layout (generate one board with visible gutters, not independent strips)"
        )
        raise StoryboardContractError(reasons, diagnostics)

    slot_width = width / columns
    slot_height = height / rows
    margin_x = max(2.0, slot_width * 0.01)
    margin_y = max(2.0, slot_height * 0.01)
    slots: dict[tuple[int, int], Component] = {}
    for component in components:
        left, top, right, bottom = component.box
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        column = int(center_x // slot_width)
        row = int(center_y // slot_height)
        if not (0 <= row < rows and 0 <= column < columns):
            reasons.append(
                f"component_label_{component.label}_center_outside_{columns}x{rows}_grid:center=({center_x:.1f},{center_y:.1f})"
            )
            continue
        slot_left, slot_top, slot_right, slot_bottom = source_slot_bounds(
            row, column, (width, height), rows=rows, columns=columns
        )
        if left <= slot_left + margin_x or right >= slot_right - margin_x or top <= slot_top + margin_y or bottom >= slot_bottom - margin_y:
            reasons.append(
                "component_label_"
                f"{component.label}_touches_or_crosses_storyboard_slot_{row},{column}:box={component.box}; "
                "add clear gutters around every complete cat"
            )
        if left <= 0 or top <= 0 or right >= width or bottom >= height:
            reasons.append(
                f"component_label_{component.label}_touches_source_canvas_edge:box={component.box}; regenerate uncropped"
            )
        key = (row, column)
        if key in slots:
            reasons.append(
                f"two_components_map_to_storyboard_slot_{row},{column}:labels={slots[key].label},{component.label}"
            )
        else:
            component.slot_row = row
            component.slot_column = column
            slots[key] = component

    missing = [f"{row},{column}" for row in range(rows) for column in range(columns) if (row, column) not in slots]
    if missing:
        reasons.append("missing_storyboard_slots:" + ";".join(missing))

    areas = np.array([component.area for component in components], dtype=np.float64)
    median_area = float(np.median(areas))
    if median_area <= 0:
        reasons.append("storyboard_has_no_recoverable_foreground_area")
    else:
        outliers = [
            component.label
            for component in components
            if component.area < median_area * 0.35 or component.area > median_area * 2.50
        ]
        if outliers:
            reasons.append(
                "component_area_outlier_labels="
                + ",".join(map(str, outliers))
                + " (each slot must contain one complete similarly scaled cat)"
            )

    if not missing:
        first = slots[(0, 0)]
        last = slots[(rows - 1, columns - 1)]

        def endpoint_ratio(first_value: float, last_value: float) -> float:
            if first_value <= 0 or last_value <= 0:
                return float("inf")
            return max(first_value, last_value) / min(first_value, last_value)

        endpoint_scale = {
            "first_slot": [0, 0],
            "last_slot": [rows - 1, columns - 1],
            "first_box": list(first.box),
            "last_box": list(last.box),
            "width_ratio": endpoint_ratio(first.width, last.width),
            "height_ratio": endpoint_ratio(first.height, last.height),
            "area_ratio": endpoint_ratio(first.area, last.area),
            "maximum_dimension_ratio": MAX_NEUTRAL_ENDPOINT_DIMENSION_RATIO,
            "maximum_area_ratio": MAX_NEUTRAL_ENDPOINT_AREA_RATIO,
        }
        diagnostics["neutral_endpoint_scale"] = endpoint_scale
        if endpoint_scale["width_ratio"] > MAX_NEUTRAL_ENDPOINT_DIMENSION_RATIO:
            reasons.append(
                "neutral_endpoint_width_scale_drift:"
                f"ratio={endpoint_scale['width_ratio']:.4f}; maximum={MAX_NEUTRAL_ENDPOINT_DIMENSION_RATIO:.4f}"
            )
        if endpoint_scale["height_ratio"] > MAX_NEUTRAL_ENDPOINT_DIMENSION_RATIO:
            reasons.append(
                "neutral_endpoint_height_scale_drift:"
                f"ratio={endpoint_scale['height_ratio']:.4f}; maximum={MAX_NEUTRAL_ENDPOINT_DIMENSION_RATIO:.4f}"
            )
        if endpoint_scale["area_ratio"] > MAX_NEUTRAL_ENDPOINT_AREA_RATIO:
            reasons.append(
                "neutral_endpoint_area_scale_drift:"
                f"ratio={endpoint_scale['area_ratio']:.4f}; maximum={MAX_NEUTRAL_ENDPOINT_AREA_RATIO:.4f}"
            )

    if reasons:
        diagnostics["slot_size"] = [slot_width, slot_height]
        raise StoryboardContractError(reasons, diagnostics)

    ordered = [slots[(row, column)] for row in range(rows) for column in range(columns)]
    return StoryboardAnalysis(
        path=path,
        raw_size=(width, height),
        keyed=keyed,
        labels=labels,
        components=ordered,
        flat_background=flat_background,
        slot_size=(slot_width, slot_height),
        rows=rows,
        columns=columns,
    )


def crop_component(analysis: StoryboardAnalysis, component: Component) -> Image.Image:
    """Extract exactly one source component and clear everything else."""

    left, top, right, bottom = component.box
    source = np.asarray(analysis.keyed, dtype=np.uint8)
    crop = source[top:bottom, left:right].copy()
    keep = analysis.labels[top:bottom, left:right] == component.label
    crop[~keep, 3] = 0
    crop[crop[..., 3] == 0, :3] = 0
    return Image.fromarray(crop, "RGBA")


def _remove_tiny_resampling_islands(array: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Remove only tiny detached alpha artifacts introduced by resampling.

    The source gate already requires exactly one connected component per cat.
    Therefore an output island of only a few pixels is a raster artifact, not a
    second valid body part.  Anything larger is deliberately retained so the
    normal one-component gate can reject the source/scale combination instead
    of silently joining or reshaping anatomy.
    """

    output = array.copy()
    output[output[..., 3] <= VISIBLE_ALPHA, 3] = 0
    mask = output[..., 3] > VISIBLE_ALPHA
    labels, components = connected_components(mask)
    removed: list[dict[str, int]] = []
    if len(components) > 1:
        primary = max(components, key=lambda component: (component.area, -component.label))
        for component in components:
            if component.label == primary.label:
                continue
            if component.area <= MAX_TINY_RESAMPLING_ISLAND_PIXELS:
                output[labels == component.label, 3] = 0
                removed.append({"label": component.label, "area": component.area})
    output[output[..., 3] == 0, :3] = 0
    return output, removed


def resize_premultiplied(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, dict[str, Any]]:
    """Resize with non-ringing alpha and no hidden RGB edge halo."""

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3]
    premultiplied = np.rint(rgba[..., :3] * alpha[..., None] / 255.0).astype(np.uint8)
    resized_rgb = np.asarray(
        Image.fromarray(premultiplied, "RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32
    )
    resized_alpha = np.asarray(
        Image.fromarray(alpha.astype(np.uint8), "L").resize(size, ALPHA_RESAMPLER), dtype=np.float32
    )
    output = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    visible = resized_alpha > 0
    restored_rgb = np.zeros_like(resized_rgb)
    restored_rgb[visible] = np.clip(
        resized_rgb[visible] * 255.0 / resized_alpha[visible, None], 0, 255
    )
    output[..., :3] = np.rint(restored_rgb).astype(np.uint8)
    output[..., 3] = np.rint(resized_alpha).astype(np.uint8)
    output, removed_islands = _remove_tiny_resampling_islands(output)
    return Image.fromarray(output, "RGBA"), {
        "color_resampler": "LANCZOS_premultiplied",
        "alpha_resampler": "BILINEAR_nonringing",
        "alpha_visibility_threshold_zeroed": VISIBLE_ALPHA,
        "tiny_resampling_islands_removed": removed_islands,
        "tiny_resampling_island_limit_pixels": MAX_TINY_RESAMPLING_ISLAND_PIXELS,
    }


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _scaled_dimension(length: int, scale: float) -> int:
    return max(1, int(math.floor(length * scale + 1e-9)))


def compute_shared_scale(analysis: StoryboardAnalysis) -> tuple[float, dict[str, Any]]:
    """Find one scale that fits all source poses and their intended x offsets."""

    limits: list[tuple[str, float]] = []
    for frame_index, component in enumerate(analysis.components):
        if component.width <= 0 or component.height <= 0:
            raise StoryboardContractError([f"frame_F{frame_index:02d}_has_empty_component_box"])
        limits.append((f"F{frame_index:02d}.height", CONTENT_HEIGHT / component.height))
        slot_left, _, slot_right, _ = source_slot_bounds(
            component.slot_row,
            component.slot_column,
            analysis.raw_size,
            rows=analysis.rows,
            columns=analysis.columns,
        )
        slot_center = (slot_left + slot_right) / 2.0
        relative_left = component.box[0] - slot_center
        relative_right = component.box[2] - slot_center
        # Preserve source-local horizontal placement while staying inside the
        # 8px side padding.  This allows a deliberate little walk without
        # giving the preparation code permission to individually recenter a
        # frame that was drawn incorrectly.
        if relative_left < 0:
            limits.append((f"F{frame_index:02d}.left_offset", (CELL_WIDTH / 2 - LEFT_PADDING) / -relative_left))
        if relative_right > 0:
            limits.append((f"F{frame_index:02d}.right_offset", (CELL_WIDTH / 2 - RIGHT_PADDING) / relative_right))
    scale = min(limit for _, limit in limits)
    if not math.isfinite(scale) or scale <= 0:
        raise StoryboardContractError(["unable_to_compute_positive_shared_scale_for_storyboard"])
    binding_name, binding_limit = min(limits, key=lambda item: item[1])
    return scale, {
        "height_or_horizontal_limit": float(binding_limit),
        "binding_constraint": binding_name,
        "content_width": float(CONTENT_WIDTH),
        "content_height": float(CONTENT_HEIGHT),
    }


def cell_metrics(cell: Image.Image) -> dict[str, Any]:
    array = np.asarray(cell.convert("RGBA"), dtype=np.uint8)
    mask = array[..., 3] > VISIBLE_ALPHA
    box = alpha_box(mask)
    transparent_rgb_pixels = int(np.count_nonzero(np.any(array[..., :3] != 0, axis=2) & (array[..., 3] == 0)))
    labels, components = connected_components(mask)
    del labels  # The record needs only deterministic component statistics.
    if box is None:
        return {
            "blank": True,
            "area": 0,
            "component_count": 0,
            "transparent_rgb_pixels": transparent_rgb_pixels,
            "key_spill_pixels": visible_key_spill_count(cell),
        }
    left, top, right, bottom = box
    ys, xs = np.where(mask)
    return {
        "blank": False,
        "box": [left, top, right, bottom],
        "width": right - left,
        "height": bottom - top,
        "area": int(mask.sum()),
        "centroid": [round(float(xs.mean()), 6), round(float(ys.mean()), 6)],
        "baseline_y": bottom,
        "padding": {
            "left": left,
            "top": top,
            "right": CELL_WIDTH - right,
            "bottom": CELL_HEIGHT - bottom,
        },
        "component_count": len(components),
        "transparent_rgb_pixels": transparent_rgb_pixels,
        "key_spill_pixels": visible_key_spill_count(cell),
    }


def require_valid_neutral(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    raw = Image.open(path)
    if raw.mode != "RGBA":
        raise ValueError(f"neutral_must_be_RGBA_not_{raw.mode}")
    neutral = raw.copy()
    if neutral.size != (CELL_WIDTH, CELL_HEIGHT):
        raise ValueError(
            f"neutral_must_be_{CELL_WIDTH}x{CELL_HEIGHT}_not_{neutral.width}x{neutral.height}"
        )
    metrics = cell_metrics(neutral)
    errors: list[str] = []
    if metrics["blank"]:
        errors.append("neutral_is_blank")
    else:
        if metrics["component_count"] != 1:
            errors.append(f"neutral_component_count={metrics['component_count']}; expected 1")
        if metrics["baseline_y"] != BASELINE_Y:
            errors.append(f"neutral_baseline={metrics['baseline_y']}; expected {BASELINE_Y}")
        padding = metrics["padding"]
        if padding["left"] < LEFT_PADDING or padding["right"] < RIGHT_PADDING or padding["top"] < TOP_PADDING:
            errors.append(f"neutral_padding_invalid:{padding}")
    if metrics["transparent_rgb_pixels"]:
        errors.append("neutral_has_nonzero_RGB_under_transparency; clean it before exact-anchor use")
    if metrics["key_spill_pixels"]:
        errors.append(f"neutral_has_magenta_key_spill={metrics['key_spill_pixels']}")
    if errors:
        raise ValueError("; ".join(errors))
    return neutral, metrics


def render_generated_cell(
    analysis: StoryboardAnalysis,
    component: Component,
    scale: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply the one approved global transform to one complete source cat."""

    crop = crop_component(analysis, component)
    target_size = (_scaled_dimension(component.width, scale), _scaled_dimension(component.height, scale))
    scaled, resampling = resize_premultiplied(crop, target_size)
    scaled_array = np.asarray(scaled, dtype=np.uint8)
    scaled_box = alpha_box(scaled_array[..., 3] > VISIBLE_ALPHA)
    if scaled_box is None:
        raise StoryboardContractError(
            [f"component_label_{component.label}_became_blank_after_shared_scale; source edge is too thin"]
        )
    slot_left, _, slot_right, _ = source_slot_bounds(
        component.slot_row,
        component.slot_column,
        analysis.raw_size,
        rows=analysis.rows,
        columns=analysis.columns,
    )
    source_slot_center = (slot_left + slot_right) / 2.0
    source_box_center = (component.box[0] + component.box[2]) / 2.0
    source_offset_x = source_box_center - source_slot_center
    target_center_x = CELL_WIDTH / 2.0 + source_offset_x * scale
    x = _round_half_up(target_center_x - (scaled_box[0] + scaled_box[2]) / 2.0)
    y = BASELINE_Y - scaled_box[3]

    cell = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    cell.alpha_composite(scaled, (x, y))
    cell = clear_transparent_rgb(cell)
    metrics = cell_metrics(cell)
    if metrics["blank"]:
        raise StoryboardContractError([f"component_label_{component.label}_is_blank_after_registration"])
    padding = metrics["padding"]
    geometry_errors: list[str] = []
    if metrics["baseline_y"] != BASELINE_Y:
        geometry_errors.append(f"registered_baseline={metrics['baseline_y']}; expected {BASELINE_Y}")
    if padding["left"] < LEFT_PADDING or padding["right"] < RIGHT_PADDING or padding["top"] < TOP_PADDING:
        geometry_errors.append(f"registered_padding={padding}; required left/right>={LEFT_PADDING}, top>={TOP_PADDING}")
    if metrics["component_count"] != 1:
        geometry_errors.append(f"registered_component_count={metrics['component_count']}; expected 1")
    if metrics["key_spill_pixels"]:
        geometry_errors.append(f"registered_key_spill={metrics['key_spill_pixels']}")
    if geometry_errors:
        raise StoryboardContractError(
            [f"component_label_{component.label}_cannot_use_one_shared_transform:" + "; ".join(geometry_errors)]
        )
    transform = {
        "kind": "complete_storyboard_cat_shared_scale",
        "scale": scale,
        "source_crop_size": [component.width, component.height],
        "scaled_size": list(target_size),
        "scaled_visible_box": list(scaled_box),
        "resampling": resampling,
        "source_slot_offset_x": round(source_offset_x, 6),
        "target_position": [x, y],
        "baseline_y": BASELINE_Y,
        "per_frame_fit": False,
        "face_overlay": False,
        "body_lock": False,
    }
    return cell, transform


def build_atlas(
    analysis: StoryboardAnalysis,
    neutral: Image.Image,
    scale: float,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Build the production atlas in strict row-major source order."""

    atlas = Image.new("RGBA", analysis.atlas_size, (0, 0, 0, 0))
    records: list[dict[str, Any]] = []
    for frame_index, component in enumerate(analysis.components):
        atlas_row, atlas_column = divmod(frame_index, analysis.columns)
        if frame_index in (0, analysis.frame_count - 1):
            cell = neutral.copy()
            transform: dict[str, Any] = {
                "kind": "exact_runtime_neutral_anchor",
                "scale": None,
                "baseline_y": BASELINE_Y,
                "per_frame_fit": False,
                "face_overlay": False,
                "body_lock": False,
            }
        else:
            cell, transform = render_generated_cell(analysis, component, scale)
        metrics = cell_metrics(cell)
        atlas.alpha_composite(cell, (atlas_column * CELL_WIDTH, atlas_row * CELL_HEIGHT))
        records.append(
            {
                "index": frame_index,
                "frame": f"F{frame_index:02d}",
                "atlas_slot": [atlas_row, atlas_column],
                "source_frame_index": frame_index,
                "storyboard_slot": [component.slot_row, component.slot_column],
                "source_component_label": component.label,
                "source_box": list(component.box),
                "source_area": component.area,
                "source_centroid": [round(component.centroid[0], 6), round(component.centroid[1], 6)],
                "transform": transform,
                "metrics": metrics,
            }
        )
    # Do not silently modify an exact neutral anchor here.  The neutral gate
    # above ensures this assertion holds instead.
    atlas_array = np.asarray(atlas, dtype=np.uint8)
    if np.any(np.any(atlas_array[..., :3] != 0, axis=2) & (atlas_array[..., 3] == 0)):
        raise RuntimeError("internal_error: atlas has hidden RGB after registration")
    return atlas, records


def write_contact_sheet(
    path: Path,
    atlas: Image.Image,
    records: list[dict[str, Any]],
    *,
    rows: int,
    columns: int,
) -> None:
    """Create a compact visual review sheet with baselines and measured boxes."""

    title_height = 25
    sheet = Image.new(
        "RGB",
        (CELL_WIDTH * columns, (CELL_HEIGHT + title_height) * rows),
        (47, 48, 52),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x = column * CELL_WIDTH
        y = row * (CELL_HEIGHT + title_height)
        cell = atlas.crop((column * CELL_WIDTH, row * CELL_HEIGHT, (column + 1) * CELL_WIDTH, (row + 1) * CELL_HEIGHT))
        tile = Image.new("RGBA", cell.size, (47, 48, 52, 255))
        tile.alpha_composite(cell)
        sheet.paste(tile.convert("RGB"), (x, y + title_height))
        metrics = record["metrics"]
        if not metrics["blank"]:
            left, top, right, bottom = metrics["box"]
            draw.rectangle(
                (x + left, y + title_height + top, x + right - 1, y + title_height + bottom - 1),
                outline=(72, 209, 222),
                width=1,
            )
        draw.line(
            (x, y + title_height + BASELINE_Y, x + CELL_WIDTH - 1, y + title_height + BASELINE_Y),
            fill=(249, 206, 74),
            width=1,
        )
        anchor = "A" if index in (0, len(records) - 1) else ""
        label = f"F{index:02d}{anchor} {metrics.get('width', 0)}x{metrics.get('height', 0)}"
        draw.text((x + 3, y + 5), label, fill=(255, 255, 255), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_failure_report(
    path: Path,
    *,
    storyboard: Path,
    neutral: Path,
    rows: int,
    columns: int,
    error: Exception,
) -> None:
    reasons = error.reasons if isinstance(error, StoryboardContractError) else [str(error)]
    diagnostics = error.diagnostics if isinstance(error, StoryboardContractError) else {}
    report = {
        "schema": "desktop-pet.interaction-v5.prepare.v1",
        "ok": False,
        "storyboard": str(storyboard),
        "runtime_neutral": str(neutral),
        "requested_grid": {"rows": rows, "columns": columns, "frame_count": rows * columns},
        "failures": reasons,
        "diagnostics": diagnostics,
        "repair_path": (
            "Regenerate one single flat-magenta storyboard in the explicitly requested 8x8/64 or 8x6/48 "
            "layout, with isolated uncropped full-cat silhouettes and clear gutters. Do not supply four strips, "
            "face patches, or a pre-sliced atlas; then rerun this preparation command."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one explicit 8x8/64 or 8x6/48 Baomihua interaction storyboard with one shared transform."
    )
    parser.add_argument("--storyboard", required=True, type=Path, help="One flat-magenta source board with one separated cat per explicit slot.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Storyboard rows: 8 (default) or explicitly 6 for a 48-frame 8x6 board.")
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS, help="Storyboard columns: currently 8.")
    parser.add_argument("--runtime-neutral", required=True, type=Path, help="Exact 192x208 neutral used only for first/final anchors.")
    parser.add_argument("--output", required=True, type=Path, help="Output RGBA atlas: width=columns*192, height=rows*208.")
    parser.add_argument("--contact", required=True, type=Path, help="Output QA contact-sheet PNG.")
    parser.add_argument("--report", required=True, type=Path, help="Output transformation and metric JSON report.")
    parser.add_argument("--frames-dir", type=Path, help="Optional directory for F00.png through the final frame QA extracts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        require_supported_grid(args.rows, args.columns)
        analysis = analyze_storyboard(args.storyboard, rows=args.rows, columns=args.columns)
        neutral, neutral_metrics = require_valid_neutral(args.runtime_neutral)
        scale, scale_constraints = compute_shared_scale(analysis)
        atlas, frames = build_atlas(analysis, neutral, scale)

        if atlas.size != analysis.atlas_size or atlas.mode != "RGBA":
            raise RuntimeError(
                f"internal_error: atlas geometry is {atlas.size} {atlas.mode}, expected {analysis.atlas_size} RGBA"
            )
        neutral_array = np.asarray(neutral, dtype=np.uint8)
        atlas_array = np.asarray(atlas, dtype=np.uint8)
        first = atlas_array[:CELL_HEIGHT, :CELL_WIDTH]
        last = atlas_array[
            (analysis.rows - 1) * CELL_HEIGHT : analysis.rows * CELL_HEIGHT,
            (analysis.columns - 1) * CELL_WIDTH : analysis.columns * CELL_WIDTH,
        ]
        if not np.array_equal(first, neutral_array) or not np.array_equal(last, neutral_array):
            raise RuntimeError("internal_error: first/final frames are not pixel-identical neutral anchors")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(args.output)
        if args.frames_dir is not None:
            args.frames_dir.mkdir(parents=True, exist_ok=True)
            for index in range(analysis.frame_count):
                row, column = divmod(index, analysis.columns)
                atlas.crop(
                    (column * CELL_WIDTH, row * CELL_HEIGHT, (column + 1) * CELL_WIDTH, (row + 1) * CELL_HEIGHT)
                ).save(args.frames_dir / f"F{index:02d}.png")
        write_contact_sheet(args.contact, atlas, frames, rows=analysis.rows, columns=analysis.columns)

        report = {
            "schema": "desktop-pet.interaction-v5.prepare.v1",
            "ok": True,
            "storyboard": str(args.storyboard),
            "storyboard_sha256": sha256(args.storyboard),
            "storyboard_size": list(analysis.raw_size),
            "runtime_neutral": str(args.runtime_neutral),
            "runtime_neutral_sha256": sha256(args.runtime_neutral),
            "neutral_metrics": neutral_metrics,
            "output": str(args.output),
            "output_sha256": sha256(args.output),
            "contact_sheet": str(args.contact),
            "atlas_geometry": {
                "size": list(analysis.atlas_size),
                "mode": "RGBA",
                "columns": analysis.columns,
                "rows": analysis.rows,
                "cell_size": [CELL_WIDTH, CELL_HEIGHT],
                "frame_count": analysis.frame_count,
            },
            "source_contract": {
                "kind": f"one_flat_magenta_{analysis.columns}x{analysis.rows}_storyboard",
                "component_count": analysis.frame_count,
                "component_order": "row_major_storyboard_slots_to_row_major_atlas_slots",
                "flat_background": analysis.flat_background,
                "source_slot_size": [round(analysis.slot_size[0], 6), round(analysis.slot_size[1], 6)],
            },
            "shared_transform": {
                "scale": scale,
                "scale_constraints": scale_constraints,
                "shared_baseline_y": BASELINE_Y,
                "content_limits": [CONTENT_WIDTH, CONTENT_HEIGHT],
                "required_padding": {
                    "left": LEFT_PADDING,
                    "right": RIGHT_PADDING,
                    "top": TOP_PADDING,
                    "bottom": BOTTOM_PADDING,
                },
                "preserve_source_horizontal_offset": True,
                "per_frame_fit": False,
                "face_overlay": False,
                "body_lock": False,
                "neutral_anchor_indices": [0, analysis.frame_count - 1],
                "neutral_may_not_be_duplicated_elsewhere": True,
            },
            "frame_order": [
                {
                    "index": record["index"],
                    "frame": record["frame"],
                    "source_frame_index": record["source_frame_index"],
                    "storyboard_slot": record["storyboard_slot"],
                    "atlas_slot": record["atlas_slot"],
                }
                for record in frames
            ],
            "frames": frames,
        }
        args.report.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(args.output),
                    "contact": str(args.contact),
                    "report": str(args.report),
                    "frame_count": analysis.frame_count,
                    "shared_scale": round(scale, 8),
                }
            )
        )
    except (StoryboardContractError, ValueError, OSError, RuntimeError) as error:
        write_failure_report(
            args.report,
            storyboard=args.storyboard,
            neutral=args.runtime_neutral,
            rows=args.rows,
            columns=args.columns,
            error=error,
        )
        print(f"prepare-interaction-v5 rejected input: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
