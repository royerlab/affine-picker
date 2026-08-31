#!/usr/bin/env python
"""Load two Zarr views into napari and wire up the point picker for registration.

Channel 0 is used as the fixed view and channel 2 as the moving view (Y-flipped)
by default. Pass ``--zarr-path`` to load both views from the same OME-Zarr, or
``--fixed-zarr-path`` / ``--moving-zarr-path`` to pull the two views from
different arrays (e.g. when aligning separate cameras).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import dask.array as da
import napari
import numpy as np

# Allow running from a source checkout without installing the plugin first.
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from napari_orthogonal_views import (  # noqa: E402
    estimate_affine_from_points_components,
    estimate_affine_from_points_masked,
    show_point_picker,
)


def _channel_label(zarr_path: Path, channel: int) -> str:
    attrs_path = zarr_path.parent / ".zattrs"
    if not attrs_path.exists():
        return f"channel {channel}"

    with attrs_path.open() as file:
        attrs = json.load(file)

    channels = attrs.get("omero", {}).get("channels", [])
    if channel >= len(channels):
        return f"channel {channel}"

    return channels[channel].get("label") or f"channel {channel}"


@dataclass(frozen=True)
class PipelineSlot:
    old_affine: np.ndarray
    post_affine: np.ndarray
    pair_offset_affine: np.ndarray
    post_processing_affine: np.ndarray
    prev_affine: np.ndarray
    old_source: str
    post_source: str
    prev_source: str
    output_target: str


@dataclass(frozen=True)
class TransformInference:
    fixed_camera_id: int | None
    moving_camera_id: int | None
    fixed_view_id: int
    moving_view_id: int
    fixed_slot: PipelineSlot | None
    moving_slot: PipelineSlot
    output_mode: str
    invert_estimated_affine: bool
    output_inverse: bool
    show_full_pipeline_matrices: bool
    shared_space_offset_mode: str
    fixed_display_affine: np.ndarray
    moving_display_affine: np.ndarray
    fixed_view_source: str
    moving_view_source: str
    # top-bottom modes: slot of the moving view's pair partner (updated with
    # the same output-space correction as the moving view).
    moving_pair_slot: PipelineSlot | None = None
    moving_pair_view_id: int | None = None
    # top-bottom-rebased: slots describing the dataset the points were
    # picked on (old transforms/offsets), used to rebase the estimated
    # affine into the current (new-transform) frame.
    points_fixed_slot: PipelineSlot | None = None
    points_moving_slot: PipelineSlot | None = None


_VIEW_LABEL_RE = re.compile(r"\bview[_\s-]*(\d+)\b", re.IGNORECASE)
_CAMERA_LABEL_RE = re.compile(r"\bcamera\s+([^\s@]+)", re.IGNORECASE)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read project transform metadata. "
            "Install pyyaml or omit --project-folder for raw-data mode."
        ) from exc

    with path.open() as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _view_key_to_int(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    text = str(key)
    try:
        return int(text)
    except ValueError:
        pass

    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else None


def _coerce_view_transforms(value: Any) -> dict[int, np.ndarray]:
    if not isinstance(value, dict):
        return {}

    transforms: dict[int, np.ndarray] = {}
    for key, matrix in value.items():
        view_id = _view_key_to_int(key)
        if view_id is None:
            continue

        array = np.asarray(matrix, dtype=float)
        if array.shape == (4, 4):
            transforms[view_id] = array

    return transforms


def _view_transforms_from_yaml(data: dict[str, Any]) -> dict[int, np.ndarray]:
    candidates = [
        data.get("view_transforms"),
        data.get("processing", {}).get("view_transforms")
        if isinstance(data.get("processing"), dict)
        else None,
    ]

    transforms: dict[int, np.ndarray] = {}
    for candidate in candidates:
        transforms.update(_coerce_view_transforms(candidate))
    return transforms


def _identity4() -> np.ndarray:
    return np.eye(4)


def _resolve_project_yaml(project_folder: Path, filename: str) -> Path | None:
    project_folder = project_folder.expanduser()
    if project_folder.is_file():
        return project_folder if project_folder.name == filename else None

    path = project_folder / filename
    return path if path.exists() else None


def _load_project_yaml(
    project_folder: Path | None,
    filename: str,
) -> tuple[dict[str, Any], str]:
    if project_folder is None:
        return {}, f"no --project-folder/{filename}"

    path = _resolve_project_yaml(project_folder, filename)
    if path is None:
        return {}, f"{project_folder}/{filename} not found"

    return _load_yaml(path), str(path)


def _project_alignment_data(project_folder: Path | None) -> tuple[dict[str, Any], str]:
    alignment_data, alignment_source = _load_project_yaml(
        project_folder,
        "views_alignment.yaml",
    )
    if alignment_data:
        return alignment_data, alignment_source

    settings_data, settings_source = _load_project_yaml(
        project_folder,
        "impp_settings.yaml",
    )
    maybe_alignment = settings_data.get("alignment")
    if isinstance(maybe_alignment, dict):
        return maybe_alignment, f"{settings_source} alignment"

    return {}, alignment_source


def _project_view_transforms(
    project_folder: Path | None,
    transform_yaml: Path | None,
) -> tuple[dict[int, np.ndarray], str]:
    if transform_yaml is not None:
        transform_yaml = transform_yaml.expanduser()
        transforms = _view_transforms_from_yaml(_load_yaml(transform_yaml))
        if transforms:
            return transforms, f"{transform_yaml} view_transforms"
        return {}, f"no view transforms found in {transform_yaml}"

    settings_data, settings_source = _load_project_yaml(
        project_folder,
        "impp_settings.yaml",
    )
    transforms = _view_transforms_from_yaml(settings_data)
    if transforms:
        return transforms, f"{settings_source} processing.view_transforms"

    alignment_data, alignment_source = _load_project_yaml(
        project_folder,
        "views_alignment.yaml",
    )
    transforms = _view_transforms_from_yaml(alignment_data)
    if transforms:
        return transforms, f"{alignment_source} view_transforms"

    return {}, f"no view transforms found in {project_folder}"


def _infer_view_id(
    explicit_view_id: int | None,
    zarr_path: Path,
    channel: int,
    preferred_view_ids: set[int] | None = None,
) -> tuple[int, str]:
    if explicit_view_id is not None:
        return explicit_view_id, "command line"

    label = _channel_label(zarr_path, channel)
    match = _VIEW_LABEL_RE.search(label)
    label_view_id = int(match.group(1)) if match else None

    if preferred_view_ids and channel in preferred_view_ids:
        if label_view_id is not None and label_view_id != channel:
            return (
                channel,
                f"channel index; label {label!r} parsed as camera-local "
                f"view {label_view_id}",
            )
        return channel, "channel index matches project view_transforms"

    if match:
        return label_view_id, f"channel label {label!r}"

    return channel, "channel index fallback"


def _project_camera_ids_by_name(project_folder: Path | None) -> dict[str, int]:
    settings_data, _ = _load_project_yaml(project_folder, "impp_settings.yaml")
    cameras = settings_data.get("cameras", {})
    if not isinstance(cameras, dict):
        return {}

    camera_ids: dict[str, int] = {}
    for key, camera in cameras.items():
        if not isinstance(camera, dict):
            continue
        name = camera.get("name")
        if not name:
            continue
        camera_id = _view_key_to_int(key)
        if camera_id is not None:
            camera_ids[str(name)] = camera_id
    return camera_ids


def _infer_camera_id(
    explicit_camera_id: int | None,
    zarr_path: Path,
    channel: int,
    project_folder: Path | None,
) -> tuple[int | None, str]:
    if explicit_camera_id is not None:
        return explicit_camera_id, "command line"

    camera_ids_by_name = _project_camera_ids_by_name(project_folder)
    if not camera_ids_by_name:
        return None, "not inferred"

    label = _channel_label(zarr_path, channel)
    match = _CAMERA_LABEL_RE.search(label)
    if not match:
        return None, f"no camera name in channel label {label!r}"

    camera_name = match.group(1)
    if camera_name not in camera_ids_by_name:
        return None, f"camera {camera_name!r} not found in project cameras"

    return (
        camera_ids_by_name[camera_name],
        f"channel label {label!r} via impp_settings.yaml cameras",
    )


def _embed_2d_transform_in_4x4(transform_2d: np.ndarray) -> np.ndarray:
    array = np.asarray(transform_2d, dtype=float)
    if array.shape != (3, 3):
        return _identity4()

    transform = _identity4()
    transform[1, 1] = array[0, 0]
    transform[1, 2] = array[0, 1]
    transform[1, 3] = array[0, 2]
    transform[2, 1] = array[1, 0]
    transform[2, 2] = array[1, 1]
    transform[2, 3] = array[1, 2]
    return transform


def _translation_affine(offset_zyx: Any | None) -> np.ndarray:
    transform = _identity4()
    if offset_zyx is not None:
        transform[:3, 3] = np.asarray(offset_zyx, dtype=float)
    return transform


def _flip_affine_for_axes(
    shape_zyx: tuple[int, int, int],
    axes: tuple[int, ...],
) -> np.ndarray:
    transform = _identity4()
    for axis in axes:
        if axis not in (0, 1, 2):
            continue
        transform[axis, axis] = -1.0
        transform[axis, 3] = shape_zyx[axis] - 1
    return transform


def _offset3d(values: Any) -> tuple[int, int, int] | None:
    if values is None:
        return None
    try:
        if len(values) != 3:
            return None
        return tuple(int(value) for value in values)
    except (TypeError, ValueError):
        return None


def _offset_scope(config: dict[str, Any], camera_id: int | None) -> dict[str, Any]:
    if camera_id is None:
        return config

    camera_key = f"camera_{camera_id}"
    camera_config = config.get(camera_key)
    if isinstance(camera_config, dict):
        return camera_config

    if config.get("per_camera_offsets"):
        camera_offsets = config.get("camera_offsets", {})
        if isinstance(camera_offsets, dict):
            scoped = camera_offsets.get(camera_key, {})
            if isinstance(scoped, dict):
                return scoped
        return {}

    return config


def _view_pair_for_view(view_id: int) -> tuple[int, int] | None:
    if view_id in (0, 1):
        return 0, 1
    if view_id in (2, 3):
        return 2, 3
    return None


def _view_pair_offset(
    alignment_data: dict[str, Any],
    view_id: int,
    camera_id: int | None,
) -> tuple[tuple[int, int, int] | None, str]:
    if view_id not in (1, 3):
        return None, f"view {view_id} has no post view-pair offset slot"

    view_pair = _view_pair_for_view(view_id)
    if view_pair is None:
        return None, f"view {view_id} has no view-pair offset mapping"

    scoped = _offset_scope(alignment_data, camera_id)
    view_pair_offsets = scoped.get("view_pair_offsets", {})
    if not isinstance(view_pair_offsets, dict):
        return None, "no view_pair_offsets"

    key = f"pair_{view_pair[0]}_{view_pair[1]}"
    offset = _offset3d(view_pair_offsets.get(key))
    if offset is None:
        return None, f"no {key} offset"
    return offset, f"view_pair_offsets.{key}"


# Views the deskew-fuse pipeline places into the shared frame with an extra
# Y-flip and the top/bottom offset. NOTE: the pipeline HARDCODES this set
# (Constants.OLD_FORMAT_BOTTOM_VIEWS = [2, 3] in impp new_impp/data/dataset.py
# -- get_bottom_views() ignores the per-channel Top/Bottom metadata in the
# settings), so this must stay hardcoded to mirror it, even for datasets
# whose channel names label views 0,1 as "Bottom Objective".
_BOTTOM_VIEW_IDS = frozenset({2, 3})


def _flip_axes_for_camera(value: Any, camera_id: int | None) -> tuple[int, ...]:
    if not isinstance(value, dict) or camera_id is None:
        return ()
    axes = None
    for key in (camera_id, str(camera_id)):
        if key in value:
            axes = value[key]
            break
    if axes is None:
        return ()
    try:
        return tuple(int(axis) for axis in axes if int(axis) in (0, 1, 2))
    except (TypeError, ValueError):
        return ()


def _camera_post_deskew_flip_axes(
    project_folder: Path | None,
    transform_yaml: Path | None,
    camera_id: int | None,
) -> tuple[tuple[int, ...], str]:
    """Read camera_axes_to_flip_post_deskew as the pipeline applied it.

    The deskew-fuse pipeline flips these axes on *every* view of the camera
    right after the affine resample. The point-picker POST model must mirror
    this so the solved view transforms land correctly once re-run.
    """
    settings_data, settings_source = _load_project_yaml(
        project_folder,
        "impp_settings.yaml",
    )
    processing = settings_data.get("processing")
    if isinstance(processing, dict) and (
        "camera_axes_to_flip_post_deskew" in processing
    ):
        axes = _flip_axes_for_camera(
            processing.get("camera_axes_to_flip_post_deskew"),
            camera_id,
        )
        return axes, f"{settings_source} processing.camera_axes_to_flip_post_deskew"
    if "camera_axes_to_flip_post_deskew" in settings_data:
        axes = _flip_axes_for_camera(
            settings_data.get("camera_axes_to_flip_post_deskew"),
            camera_id,
        )
        return axes, f"{settings_source} camera_axes_to_flip_post_deskew"
    if transform_yaml is not None:
        data = _load_yaml(transform_yaml.expanduser())
        axes = _flip_axes_for_camera(
            data.get("camera_axes_to_flip_post_deskew"),
            camera_id,
        )
        return axes, f"{transform_yaml} camera_axes_to_flip_post_deskew"
    return (), "no camera_axes_to_flip_post_deskew found (assuming none)"


def _top_bottom_offset(
    alignment_data: dict[str, Any],
    camera_id: int | None,
) -> tuple[tuple[int, int, int] | None, str]:
    scoped = _offset_scope(alignment_data, camera_id)
    offset = _offset3d(
        scoped.get("top_bottom_offsets") or scoped.get("offsets")
    )
    if offset is not None:
        return offset, "top_bottom_offsets"
    offset = _offset3d(
        alignment_data.get("top_bottom_offsets")
        or alignment_data.get("offsets")
    )
    if offset is not None:
        return offset, "top_bottom_offsets"
    return None, "no top_bottom_offsets"


def _top_bottom_canvas_growth(
    alignment_data: dict[str, Any],
) -> tuple[int, int, int]:
    """Return growth added after per-view flips in deskew-only output.

    The processing pipeline computes the shared canvas by adding the positive
    part of the largest top/bottom offset to the deskewed view shape.  Bottom
    views are flipped *before* that growth.  Consequently, a final Zarr shape
    cannot be used directly as the flip extent when any positive offset grew
    the canvas.
    """
    offsets: list[tuple[int, int, int]] = []

    def add_offset(scope: Any) -> None:
        if not isinstance(scope, dict):
            return
        offset = _offset3d(
            scope.get("top_bottom_offsets") or scope.get("offsets")
        )
        if offset is not None:
            offsets.append(offset)

    per_camera = bool(alignment_data.get("per_camera_offsets"))
    if not per_camera:
        add_offset(alignment_data)

    camera_offsets = alignment_data.get("camera_offsets")
    if isinstance(camera_offsets, dict):
        for scope in camera_offsets.values():
            add_offset(scope)

    # Also support the legacy top-level camera_0/camera_1 schema.  These
    # scopes take precedence in the processing pipeline.
    for key, scope in alignment_data.items():
        if str(key).startswith("camera_"):
            add_offset(scope)

    return tuple(
        max([0, *(offset[axis] for offset in offsets)])
        for axis in range(3)
    )


def _pre_top_bottom_view_shape(
    output_shape_zyx: tuple[int, int, int],
    alignment_data: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Recover the per-view shape at which post-deskew flips are applied."""
    growth = _top_bottom_canvas_growth(alignment_data)
    view_shape = tuple(
        int(output_shape_zyx[axis]) - growth[axis] for axis in range(3)
    )
    if any(size <= 0 for size in view_shape):
        raise SystemExit(
            "Top/bottom canvas growth is incompatible with the output shape: "
            f"shape={output_shape_zyx}, growth={growth}."
        )
    return view_shape, growth


def _camera_alignment_affine(
    alignment_data: dict[str, Any],
    camera_id: int | None,
    view_id: int,
) -> tuple[np.ndarray, str]:
    if camera_id is None:
        return _identity4(), "no camera id"

    camera_alignment = alignment_data.get("camera_alignment")
    if not isinstance(camera_alignment, dict):
        return _identity4(), "no camera_alignment"

    camera_key = f"camera_{camera_id}"
    view_key = f"view_{view_id}"
    transforms = camera_alignment.get("transforms", {})
    camera_transforms = (
        transforms.get(camera_key, {}) if isinstance(transforms, dict) else {}
    )

    if isinstance(camera_transforms, dict) and view_key in camera_transforms:
        transform_2d = np.asarray(camera_transforms[view_key], dtype=float)
        return (
            _embed_2d_transform_in_4x4(transform_2d),
            f"camera_alignment.transforms.{camera_key}.{view_key}",
        )

    if camera_alignment.get("reference_camera") == camera_id:
        return _identity4(), f"{camera_key} is camera_alignment reference"

    return _identity4(), f"no camera_alignment for {camera_key}.{view_key}"


def _view_post_processing_affine(
    view_id: int,
    shape_zyx: tuple[int, int, int],
    post_deskew_flip_axes: tuple[int, ...],
    top_bottom_offset: tuple[int, int, int] | None,
) -> tuple[np.ndarray, str]:
    """POST factor between the view affine and the shared deskew-fuse frame.

    ``shape_zyx`` is the deskewed per-view shape before the shared canvas is
    grown for top/bottom placement.  This must reproduce what the pipeline
    does after ``resample_affine`` (and before the view-pair offset, handled
    separately):

    1. Post-deskew flips on *every* view of the camera
       (``camera_axes_to_flip_post_deskew``).
    2. Top/bottom placement for the inverted-objective views (2, 3): an extra
       Y-flip plus the ``top_bottom_offset`` translation.

    For the usual DaXI config (post-deskew Y-flip on all views) the two Y-flips
    cancel on the bottom views, leaving only the top/bottom offset there, while
    the top views (0, 1) keep a single Y-flip. The previous implementation had
    this backwards -- it flipped only views 2, 3 and ignored the top/bottom
    offset -- which corrupted the solved transforms once the pipeline re-ran.
    """
    transform = _flip_affine_for_axes(shape_zyx, post_deskew_flip_axes)
    notes = [f"post-deskew flip axes={tuple(post_deskew_flip_axes)}"]

    if view_id in _BOTTOM_VIEW_IDS:
        tb_flip = _flip_affine_for_axes(shape_zyx, (1,))
        tb_shift = _translation_affine(top_bottom_offset)
        transform = tb_shift @ tb_flip @ transform
        notes.append(
            "top/bottom placement Y-flip + offset "
            f"{tuple(top_bottom_offset) if top_bottom_offset else (0, 0, 0)}"
        )

    return transform, "; ".join(notes)


def _infer_pipeline_slot(
    *,
    alignment_mode: str,
    project_folder: Path | None,
    transform_yaml: Path | None,
    view_id: int,
    camera_id: int | None,
    view_shape_zyx: tuple[int, int, int],
    allow_identity_old_transform: bool,
) -> PipelineSlot:
    if alignment_mode == "camera":
        return PipelineSlot(
            old_affine=_identity4(),
            post_affine=_identity4(),
            pair_offset_affine=_identity4(),
            post_processing_affine=_identity4(),
            prev_affine=_identity4(),
            old_source="camera alignment slot identity",
            post_source="camera alignment post identity",
            prev_source="camera alignment prev handled by loaded data",
            output_target="camera_alignment",
        )

    view_transforms, transform_source = _project_view_transforms(
        project_folder,
        transform_yaml,
    )
    if view_id not in view_transforms:
        if not allow_identity_old_transform:
            raise SystemExit(
                "View alignment mode requires the existing view transform. "
                f"Could not find view {view_id} in {transform_source}. "
                "Pass --project-folder pointing at an impp output directory "
                "with impp_settings.yaml, or pass --transform-yaml pointing "
                "at the transforms YAML. Use --allow-identity-old-transform "
                "only for raw/identity-slot debugging."
            )
        old_affine = _identity4()
        old_source = f"identity; {transform_source}"
    else:
        old_affine = view_transforms[view_id]
        old_source = f"{transform_source}[{view_id}]"

    alignment_data, alignment_source = _project_alignment_data(project_folder)
    post_offset, post_source = _view_pair_offset(
        alignment_data,
        view_id,
        camera_id,
    )
    pair_offset_affine = _translation_affine(post_offset)
    if post_offset is not None:
        post_source = f"{alignment_source} {post_source}"

    post_deskew_flip_axes, flip_source = _camera_post_deskew_flip_axes(
        project_folder,
        transform_yaml,
        camera_id,
    )
    tb_offset, tb_source = _top_bottom_offset(alignment_data, camera_id)
    pre_placement_shape, canvas_growth = _pre_top_bottom_view_shape(
        view_shape_zyx,
        alignment_data,
    )
    post_processing_affine, post_processing_source = _view_post_processing_affine(
        view_id,
        pre_placement_shape,
        post_deskew_flip_axes,
        tb_offset,
    )
    post_affine = pair_offset_affine @ post_processing_affine
    post_source = (
        f"{post_source}; {post_processing_source} "
        f"[pre-placement shape={pre_placement_shape}, canvas growth="
        f"{canvas_growth}; {flip_source}; {alignment_source} {tb_source}]"
    )

    prev_affine, prev_source = _camera_alignment_affine(
        alignment_data,
        camera_id,
        view_id,
    )
    if alignment_data:
        prev_source = f"{alignment_source} {prev_source}"

    return PipelineSlot(
        old_affine=old_affine,
        post_affine=post_affine,
        pair_offset_affine=pair_offset_affine,
        post_processing_affine=post_processing_affine,
        prev_affine=prev_affine,
        old_source=old_source,
        post_source=post_source,
        prev_source=prev_source,
        output_target="view_transforms",
    )


@dataclass(frozen=True)
class ViewFlips:
    y: bool = False
    x: bool = False

    def tags(self) -> list[str]:
        tags = []
        if self.y:
            tags.append("Y flipped")
        if self.x:
            tags.append("X flipped")
        return tags


def _display_affine_for_flips(
    shape_zyx: tuple[int, int, int],
    flips: ViewFlips,
) -> np.ndarray:
    transform = _identity4()
    if flips.y:
        transform[1, 1] = -1.0
        transform[1, 3] = shape_zyx[1] - 1
    if flips.x:
        transform[2, 2] = -1.0
        transform[2, 3] = shape_zyx[2] - 1
    return transform


def _format_yaml_matrix(matrix: np.ndarray, indent: str = "    ") -> str:
    rows = []
    for row in matrix:
        values = ", ".join(f"{value:.10f}" for value in row)
        rows.append(f"{indent}- [{values}]")
    return "\n".join(rows)


def _format_camera_alignment_3x3(matrix: np.ndarray, indent: str = "        ") -> str:
    camera_matrix = np.array(
        [
            [matrix[1, 1], matrix[1, 2], matrix[1, 3]],
            [matrix[2, 1], matrix[2, 2], matrix[2, 3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return _format_yaml_matrix(camera_matrix, indent=indent)


def _normalize_homogeneous_affine(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=float).copy()
    normalized[-1, :] = 0.0
    normalized[-1, -1] = 1.0
    return normalized


def _matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    from scipy.linalg import sqrtm

    root = sqrtm(matrix)
    root = np.real_if_close(root, tol=1000)
    if np.iscomplexobj(root):
        max_imag = float(np.max(np.abs(np.imag(root))))
        raise RuntimeError(
            "Could not compute a real square-root split for the estimated "
            f"affine; max imaginary component was {max_imag:.3e}."
        )
    return _normalize_homogeneous_affine(root)


def _solve_slot_update(
    slot: PipelineSlot,
    correction_affine: np.ndarray,
    *,
    old_post_affine: np.ndarray | None = None,
    output_post_affine: np.ndarray | None = None,
) -> np.ndarray:
    old_post = slot.post_affine if old_post_affine is None else old_post_affine
    output_post = (
        slot.post_affine if output_post_affine is None else output_post_affine
    )
    return (
        np.linalg.inv(output_post)
        @ correction_affine
        @ old_post
        @ slot.old_affine
    )


def _estimated_direction(invert_estimated_affine: bool) -> str:
    if invert_estimated_affine:
        return "fixed-to-moving"
    return "moving-to-fixed"


def _centered_post_affines(
    fixed_post_affine: np.ndarray,
    moving_post_affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_offset = np.asarray(fixed_post_affine[:3, 3], dtype=float)
    moving_offset = np.asarray(moving_post_affine[:3, 3], dtype=float)
    center_offset = 0.5 * (fixed_offset + moving_offset)

    fixed_centered = np.asarray(fixed_post_affine, dtype=float).copy()
    moving_centered = np.asarray(moving_post_affine, dtype=float).copy()
    fixed_centered[:3, 3] = fixed_offset - center_offset
    moving_centered[:3, 3] = moving_offset - center_offset
    return fixed_centered, moving_centered, center_offset


def _shared_view_pair_or_raise(
    fixed_view_id: int,
    moving_view_id: int,
) -> tuple[int, int]:
    fixed_pair = _view_pair_for_view(fixed_view_id)
    moving_pair = _view_pair_for_view(moving_view_id)
    if fixed_pair is None or moving_pair is None or fixed_pair != moving_pair:
        raise SystemExit(
            "--output-mode shared-space requires fixed and moving views from "
            "the same pair: 0/1 or 2/3. Got fixed view "
            f"{fixed_view_id} and moving view {moving_view_id}."
        )
    return fixed_pair


def _top_bottom_partner_or_raise(
    fixed_view_id: int,
    moving_view_id: int,
) -> int:
    """Validate a cross-pair (top<->bottom) setup; return the moving partner.

    The moving view and its pair partner (0<->1 or 2<->3) both receive the
    same output-space correction, so their mutual normalization is kept.
    """
    fixed_pair = _view_pair_for_view(fixed_view_id)
    moving_pair = _view_pair_for_view(moving_view_id)
    if fixed_pair is None or moving_pair is None or fixed_pair == moving_pair:
        raise SystemExit(
            "top-bottom output modes require fixed and moving views from "
            "DIFFERENT pairs (e.g. fixed view 0, moving view 2). Got fixed "
            f"view {fixed_view_id} and moving view {moving_view_id}."
        )
    partner = [v for v in moving_pair if v != moving_view_id]
    return partner[0]


def _full_pipeline_affine(slot: PipelineSlot) -> np.ndarray:
    """Raw-data -> output-frame map realized by the pipeline for a slot."""
    return slot.post_affine @ slot.old_affine @ slot.prev_affine


def _print_top_bottom_affines(
    data_affine: np.ndarray,
    inference: TransformInference,
) -> None:
    """Apply a cross-pair correction to the moving view AND its partner.

    ``data_affine`` maps moving-view output coords -> fixed-view output
    coords in the CURRENT (new-transform) frame. Both moving-pair views get
    A_new = inv(POST) @ correction @ POST @ A_old, which realigns them onto
    the fixed pair while preserving their mutual normalization and the
    existing views_alignment.yaml offsets.
    """
    moving_slot = inference.moving_slot
    pair_slot = inference.moving_pair_slot
    if pair_slot is None or inference.moving_pair_view_id is None:
        print("[TransformYAML] Missing moving-pair slot; skipping output.")
        return
    if moving_slot.output_target != "view_transforms":
        print(
            "[TransformYAML] top-bottom output is supported only for "
            "view_transforms; skipping YAML output."
        )
        return

    updates = {}
    for view_id, slot in (
        (inference.moving_view_id, moving_slot),
        (inference.moving_pair_view_id, pair_slot),
    ):
        updates[view_id] = _solve_slot_update(slot, data_affine)

    print("[TransformYAML] Top-bottom solve:")
    print(
        "[TransformYAML] correction maps moving view data -> fixed view data"
    )
    print(
        "[TransformYAML] A_new = inv(POST) @ correction @ POST @ A_old, "
        "applied to BOTH moving-pair views"
    )
    print(f"[TransformYAML] Fixed view: {inference.fixed_view_id}")
    print(
        f"[TransformYAML] Moving pair: {inference.moving_view_id} "
        f"(picked) + {inference.moving_pair_view_id} (partner)"
    )
    print(f"[TransformYAML] Moving A_old source: {moving_slot.old_source}")
    print(f"[TransformYAML] Moving POST source: {moving_slot.post_source}")
    print(f"[TransformYAML] Partner A_old source: {pair_slot.old_source}")
    print(f"[TransformYAML] Partner POST source: {pair_slot.post_source}")
    print(
        "[TransformYAML] Keep views_alignment.yaml offsets AS THEY ARE: the "
        "solve preserves the existing POST factors."
    )
    print("[TransformYAML] Copy-paste block:")
    print("view_transforms:")
    for view_id in sorted(updates):
        print(f"  {view_id}:")
        print(_format_yaml_matrix(updates[view_id]))


def _load_point_pairs_json(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load saved point-picker pairs: layer1 = fixed, layer2 = moving."""
    import json

    with path.open() as file:
        data = json.load(file)
    layer1 = data["layer1_name"]
    layer2 = data["layer2_name"]
    fixed_pts = np.asarray(data["pairs"][layer1], dtype=float)
    moving_pts = np.asarray(data["pairs"][layer2], dtype=float)
    if fixed_pts.shape != moving_pts.shape or fixed_pts.shape[1] != 3:
        raise SystemExit(
            f"{path}: expected matching (N, 3) point lists, got "
            f"{fixed_pts.shape} and {moving_pts.shape}."
        )
    return fixed_pts, moving_pts, layer1, layer2


def _warn_if_extreme(affine: np.ndarray, label: str) -> None:
    deviation = np.abs(affine[:3, :3] - np.eye(3)).max()
    if deviation > 0.35:
        print(
            f"[TransformYAML] WARNING: {label} deviates strongly from "
            f"identity (max linear deviation {deviation:.3f}). With few or "
            "near-coplanar points the estimator can pick a degenerate "
            "shear/rotation; double-check the residuals per axis."
        )


def _run_top_bottom_rebased(
    args: argparse.Namespace,
    inference: TransformInference,
    affine_estimator,
) -> None:
    """Headless top-bottom solve from a saved point-pairs JSON.

    The pairs were picked on a dataset generated with OLD moving-pair
    transforms (--points-project-folder / --points-transform-yaml). Each
    view's points are first mapped into the CURRENT frame through
    M = FULL_new @ inv(FULL_old), then the cross-pair correction is fitted
    and folded into both moving-pair views.
    """
    fixed_pts, moving_pts, layer1, layer2 = _load_point_pairs_json(
        args.point_pairs_json.expanduser()
    )
    print(f"[TopBottom] Loaded {len(fixed_pts)} pairs from {args.point_pairs_json}")
    print(f"[TopBottom] fixed layer:  {layer1}")
    print(f"[TopBottom] moving layer: {layer2}")

    if inference.points_fixed_slot is None or inference.points_moving_slot is None:
        raise SystemExit(
            "top-bottom-rebased requires --points-project-folder (or "
            "--points-transform-yaml) describing the dataset the points "
            "were picked on."
        )

    rebase = {}
    for label, new_slot, old_slot in (
        ("fixed", inference.fixed_slot, inference.points_fixed_slot),
        ("moving", inference.moving_slot, inference.points_moving_slot),
    ):
        full_new = _full_pipeline_affine(new_slot)
        full_old = _full_pipeline_affine(old_slot)
        rebase[label] = full_new @ np.linalg.inv(full_old)
        print(f"[TopBottom] {label} rebase map M = FULL_new @ inv(FULL_old):")
        print(_format_yaml_matrix(rebase[label]))
        if np.allclose(rebase[label], _identity4(), atol=1e-9):
            print(f"[TopBottom] {label} frame unchanged (M = identity).")

    def apply(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return (mat[:3, :3] @ pts.T).T + mat[:3, 3]

    fixed_new = apply(rebase["fixed"], fixed_pts)
    moving_new = apply(rebase["moving"], moving_pts)
    print("[TopBottom] Rebased points (fixed | moving), current frame:")
    for f_pt, m_pt in zip(fixed_new, moving_new, strict=True):
        print(
            f"  {np.round(f_pt, 2).tolist()} | {np.round(m_pt, 2).tolist()}"
        )
    if args.save_rebased_points is not None:
        import json

        payload = {
            "layer1_name": layer1,
            "layer2_name": layer2,
            "pairs": {
                layer1: fixed_new.tolist(),
                layer2: moving_new.tolist(),
            },
        }
        out_path = args.save_rebased_points.expanduser()
        with out_path.open("w") as file:
            json.dump(payload, file, indent=2)
        print(f"[TopBottom] Saved rebased pairs to {out_path}")

    correction = affine_estimator(moving_new, fixed_new)
    residuals = apply(correction, moving_new) - fixed_new
    rms = float(np.sqrt((residuals**2).sum(axis=1).mean()))
    print(
        f"[TopBottom] Fitted correction on {len(fixed_new)} rebased pairs: "
        f"RMS {rms:.2f} px, per-axis std "
        f"{np.round(residuals.std(axis=0), 2).tolist()}"
    )
    _warn_if_extreme(correction, "fitted correction")
    if inference.invert_estimated_affine:
        correction = np.linalg.inv(correction)
        print("[TopBottom] Inverted fitted affine before slot solve.")
    _print_top_bottom_affines(correction, inference)


def _print_yaml_ready_affine(
    manager,
    inference: TransformInference,
) -> None:
    data_affine = manager.get_estimated_affine()
    if data_affine is None:
        return

    if data_affine.shape != (4, 4):
        print(
            "[TransformYAML] Expected a 4x4 3D affine, "
            f"got {data_affine.shape}; skipping YAML output."
        )
        return

    if not (
        np.allclose(inference.fixed_display_affine, _identity4())
        and np.allclose(inference.moving_display_affine, _identity4())
    ):
        data_affine = (
            inference.fixed_display_affine
            @ data_affine
            @ np.linalg.inv(inference.moving_display_affine)
        )
        print(
            "[TransformYAML] Converted estimated affine from displayed "
            "flipped data coordinates back to original data coordinates."
        )

    if inference.invert_estimated_affine:
        data_affine = np.linalg.inv(data_affine)
        print("[TransformYAML] Inverted estimated affine before slot solve.")

    if inference.output_mode == "shared-space":
        _print_shared_space_affines(data_affine, inference)
        return

    if inference.output_mode == "top-bottom-direct":
        _warn_if_extreme(data_affine, "estimated affine")
        _print_top_bottom_affines(data_affine, inference)
        return

    slot = inference.moving_slot
    output_affine = _solve_slot_update(slot, data_affine)
    output_label = "A_new"
    if inference.output_inverse:
        output_affine = np.linalg.inv(output_affine)
        output_label = "inv(A_new)"

    print("[TransformYAML] Pipeline slot solve:")
    print(
        "[TransformYAML] estimated @ POST @ A_old @ PREV "
        "= POST @ A_new @ PREV"
    )
    print("[TransformYAML] A_new = inv(POST) @ estimated @ POST @ A_old")
    if inference.output_inverse:
        print("[TransformYAML] Emitting inverse: inv(A_new)")
    print(f"[TransformYAML] Fixed view: {inference.fixed_view_id}")
    print(f"[TransformYAML] Moving view: {inference.moving_view_id}")
    if inference.fixed_camera_id is not None:
        print(f"[TransformYAML] Fixed camera: {inference.fixed_camera_id}")
    if inference.moving_camera_id is not None:
        print(f"[TransformYAML] Moving camera: {inference.moving_camera_id}")
    print(f"[TransformYAML] A_old source: {slot.old_source}")
    print(f"[TransformYAML] POST source: {slot.post_source}")
    print(f"[TransformYAML] PREV source: {slot.prev_source}")
    print("[TransformYAML] Copy-paste block:")
    if slot.output_target == "camera_alignment":
        camera_id = (
            inference.moving_camera_id
            if inference.moving_camera_id is not None
            else "MOVING_CAMERA_ID"
        )
        print("camera_alignment:")
        print("  transforms:")
        print(f"    camera_{camera_id}:")
        print(f"      view_{inference.moving_view_id}:")
        print(_format_camera_alignment_3x3(output_affine))
        print(f"[TransformYAML] Equivalent embedded 4x4 ({output_label}):")
        print(_format_yaml_matrix(output_affine))
    else:
        print("view_transforms:")
        print(f"  {inference.moving_view_id}:")
        print(_format_yaml_matrix(output_affine))
    if inference.show_full_pipeline_matrices and not inference.output_inverse:
        full_affine = slot.post_affine @ output_affine @ slot.prev_affine
        print(
            "[TransformYAML] Diagnostic only; do not paste this into "
            "view_transforms."
        )
        print(
            "[TransformYAML] Full pipeline matrix "
            "(POST @ A_new @ PREV):"
        )
        print(_format_yaml_matrix(full_affine))


def _print_shared_space_affines(
    data_affine: np.ndarray,
    inference: TransformInference,
) -> None:
    if inference.fixed_slot is None:
        print("[TransformYAML] Missing fixed slot; skipping shared-space output.")
        return

    fixed_slot = inference.fixed_slot
    moving_slot = inference.moving_slot

    if fixed_slot.output_target != "view_transforms":
        print(
            "[TransformYAML] Shared-space output is supported only for "
            "view_transforms; skipping YAML output."
        )
        return

    view_pair = _shared_view_pair_or_raise(
        inference.fixed_view_id,
        inference.moving_view_id,
    )

    fixed_old_post = fixed_slot.post_affine
    moving_old_post = moving_slot.post_affine
    fixed_output_post = fixed_slot.post_affine
    moving_output_post = moving_slot.post_affine
    fixed_old_offset = fixed_slot.pair_offset_affine
    moving_old_offset = moving_slot.pair_offset_affine
    post_center = np.zeros(3)
    if inference.shared_space_offset_mode == "split-offset":
        fixed_old_offset, moving_old_offset, post_center = (
            _centered_post_affines(
                fixed_slot.pair_offset_affine,
                moving_slot.pair_offset_affine,
            )
        )
        fixed_old_post = fixed_old_offset @ fixed_slot.post_processing_affine
        moving_old_post = moving_old_offset @ moving_slot.post_processing_affine
        fixed_output_post = fixed_slot.post_processing_affine
        moving_output_post = moving_slot.post_processing_affine
        rebase = _translation_affine(-post_center)
        data_affine = rebase @ data_affine @ np.linalg.inv(rebase)

    half_affine = _matrix_square_root(data_affine)
    direction = _estimated_direction(inference.invert_estimated_affine)
    if direction == "moving-to-fixed":
        fixed_correction = np.linalg.inv(half_affine)
        moving_correction = half_affine
        fixed_correction_label = "inv(half)"
        moving_correction_label = "half"
        direction_label = "moving view data -> fixed view data"
    else:
        fixed_correction = half_affine
        moving_correction = np.linalg.inv(half_affine)
        fixed_correction_label = "half"
        moving_correction_label = "inv(half)"
        direction_label = "fixed view data -> moving view data"
    fixed_affine = _solve_slot_update(
        fixed_slot,
        fixed_correction,
        old_post_affine=fixed_old_post,
        output_post_affine=fixed_output_post,
    )
    moving_affine = _solve_slot_update(
        moving_slot,
        moving_correction,
        old_post_affine=moving_old_post,
        output_post_affine=moving_output_post,
    )
    output_label = "A_new/B_new"

    if inference.output_inverse:
        fixed_affine = np.linalg.inv(fixed_affine)
        moving_affine = np.linalg.inv(moving_affine)
        output_label = "inv(A_new)/inv(B_new)"

    print("[TransformYAML] Shared-space solve:")
    print(f"[TransformYAML] estimated maps {direction_label}")
    print(
        "[TransformYAML] shared-space offset mode: "
        f"{inference.shared_space_offset_mode}"
    )
    if inference.shared_space_offset_mode == "split-offset":
        print(
            "[TransformYAML] centered POST offsets around pair center "
            f"{tuple(float(axis) for axis in post_center)}"
        )
        print(
            "[TransformYAML] fixed centered POST offset: "
            f"{tuple(float(axis) for axis in fixed_old_offset[:3, 3])}"
        )
        print(
            "[TransformYAML] moving centered POST offset: "
            f"{tuple(float(axis) for axis in moving_old_offset[:3, 3])}"
        )
        print(
            "[TransformYAML] output view-pair offsets are zero; pipeline "
            "post-deskew flips remain in POST."
        )
    print("[TransformYAML] half = sqrt(estimated)")
    print(f"[TransformYAML] fixed correction = {fixed_correction_label}")
    print(f"[TransformYAML] moving correction = {moving_correction_label}")
    if inference.shared_space_offset_mode == "preserve-post":
        print(
            "[TransformYAML] correction @ POST @ A_old @ PREV "
            "= POST @ A_new @ PREV"
        )
        print(
            "[TransformYAML] A_new = inv(POST) @ correction @ POST @ A_old"
        )
    else:
        print(
            "[TransformYAML] correction @ CENTERED_POST @ A_old @ PREV "
            "= OUTPUT_POST @ A_new @ PREV"
        )
        print(
            "[TransformYAML] A_new = inv(OUTPUT_POST) @ correction "
            "@ CENTERED_POST @ A_old"
        )
    if inference.output_inverse:
        print("[TransformYAML] Emitting inverse for both view transforms.")
    print(f"[TransformYAML] Fixed view: {inference.fixed_view_id}")
    print(f"[TransformYAML] Moving view: {inference.moving_view_id}")
    if inference.fixed_camera_id is not None:
        print(f"[TransformYAML] Fixed camera: {inference.fixed_camera_id}")
    if inference.moving_camera_id is not None:
        print(f"[TransformYAML] Moving camera: {inference.moving_camera_id}")
    print(f"[TransformYAML] Fixed A_old source: {fixed_slot.old_source}")
    print(f"[TransformYAML] Fixed POST source: {fixed_slot.post_source}")
    print(f"[TransformYAML] Fixed PREV source: {fixed_slot.prev_source}")
    print(f"[TransformYAML] Moving A_old source: {moving_slot.old_source}")
    print(f"[TransformYAML] Moving POST source: {moving_slot.post_source}")
    print(f"[TransformYAML] Moving PREV source: {moving_slot.prev_source}")
    print("[TransformYAML] Copy-paste block:")
    print("view_transforms:")
    print(f"  {inference.fixed_view_id}:")
    print(_format_yaml_matrix(fixed_affine))
    print(f"  {inference.moving_view_id}:")
    print(_format_yaml_matrix(moving_affine))
    if inference.shared_space_offset_mode == "split-offset":
        print("[TransformYAML] Also update views_alignment.yaml:")
        print("view_pair_offsets:")
        print(f"  pair_{view_pair[0]}_{view_pair[1]}:")
        print("    - 0")
        print("    - 0")
        print("    - 0")
    print(f"[TransformYAML] Output label: {output_label}")
    if inference.show_full_pipeline_matrices and not inference.output_inverse:
        fixed_full = fixed_output_post @ fixed_affine @ fixed_slot.prev_affine
        moving_full = (
            moving_output_post @ moving_affine @ moving_slot.prev_affine
        )
        print(
            "[TransformYAML] Diagnostic only; do not paste these into "
            "view_transforms."
        )
        print(
            "[TransformYAML] Full pipeline matrices "
            "(POST @ output @ PREV):"
        )
        print(f"  {inference.fixed_view_id}:")
        print(_format_yaml_matrix(fixed_full))
        print(f"  {inference.moving_view_id}:")
        print(_format_yaml_matrix(moving_full))


def _build_affine_estimator(alignment_mode: str):
    if alignment_mode == "view":
        # VIEW alignment: use when aligning two views that should already share
        # the same detector/stage geometry after deskew. The estimator allows
        # Z translation and in-plane geometry, but keeps Z-dependent rotations
        # fixed so XY slices remain geometrically consistent through the stack.
        return (
            partial(
                estimate_affine_from_points_components,
                fix_rotation=(False, True, True),
                fix_scale=(True, False, False),
            ),
            "VIEW alignment estimator",
        )

    if alignment_mode == "camera":
        # CAMERA alignment: use when aligning two physical cameras from raw or
        # camera-specific views. The estimator leaves Z fixed and only solves
        # the XY plane relationship: in-plane rotation/scale/shear plus Y/X
        # translations.
        mask = np.zeros((3, 4), dtype=bool)
        mask[1:3, 1:4] = True
        return (
            partial(
                estimate_affine_from_points_masked,
                mask=mask,
            ),
            "CAMERA alignment estimator",
        )

    raise ValueError(f"Unsupported alignment mode: {alignment_mode}")


def _slice_channel(
    data: da.Array,
    *,
    timepoint: int,
    channel: int,
    flips: ViewFlips | None = None,
) -> da.Array:
    if flips is None:
        flips = ViewFlips()
    if data.ndim != 5:
        raise ValueError(
            "Expected a 5D Zarr array with axes (T, C, Z, Y, X), "
            f"but got shape {data.shape}."
        )
    if not 0 <= timepoint < data.shape[0]:
        raise ValueError(
            f"timepoint {timepoint} is out of bounds for T={data.shape[0]}"
        )
    if not 0 <= channel < data.shape[1]:
        raise ValueError(
            f"channel {channel} is out of bounds for C={data.shape[1]}"
        )

    volume = data[timepoint, channel]
    if flips.y:
        volume = volume[:, ::-1, :]
    if flips.x:
        volume = volume[:, :, ::-1]
    return volume.compute()


def _spatial_shape_from_zarr(
    zarr_path: Path,
    *,
    level_hint: str | None = None,
) -> tuple[tuple[int, int, int], str]:
    """Read a spatial shape from a Zarr array or OME-Zarr root."""
    path = zarr_path.expanduser()
    if not (path / ".zarray").exists():
        candidates = []
        if level_hint is not None:
            candidates.append(path / level_hint)
        candidates.append(path / "0")
        path = next(
            (
                candidate
                for candidate in candidates
                if (candidate / ".zarray").exists()
            ),
            path,
        )

    try:
        shape = tuple(int(size) for size in da.from_zarr(str(path)).shape)
    except Exception as exc:
        raise SystemExit(
            f"Could not read the current-frame shape from {path}: {exc}"
        ) from exc

    if len(shape) == 5:
        spatial_shape = shape[2:]
    elif len(shape) == 3:
        spatial_shape = shape
    else:
        raise SystemExit(
            f"Expected a 3D or 5D current Zarr array at {path}, got shape {shape}."
        )
    return spatial_shape, str(path)


def _infer_current_zarr_path(
    project_folder: Path | None,
) -> tuple[Path | None, str]:
    """Find the output Zarr recorded by the current project settings."""
    if project_folder is None:
        return None, "no --project-folder"

    settings_path = _resolve_project_yaml(
        project_folder.expanduser(),
        "impp_settings.yaml",
    )
    if settings_path is None:
        return None, f"no impp_settings.yaml in {project_folder}"

    settings = _load_yaml(settings_path)
    job = settings.get("job")
    output_path = job.get("output_path") if isinstance(job, dict) else None
    if not output_path:
        return None, f"no job.output_path in {settings_path}"
    return Path(output_path).expanduser(), f"{settings_path} job.output_path"


def _resolve_current_spatial_shape(
    *,
    current_shape: tuple[int, int, int] | list[int] | None,
    current_zarr_path: Path | None,
    project_folder: Path | None,
    points_zarr_path: Path,
) -> tuple[tuple[int, int, int], str]:
    """Resolve the regenerated frame shape used by current pipeline slots.

    ``points_zarr_path`` belongs to the old dataset and is used only to retain
    its pyramid-level name while resolving an OME-Zarr root. It must never be
    used as the current shape fallback: a changed Y extent changes the origin
    of the pipeline's post-resample bottom-view flip.
    """
    if current_shape is not None and current_zarr_path is not None:
        raise SystemExit(
            "Use only one of --current-shape and --current-zarr-path."
        )

    if current_shape is not None:
        shape = tuple(int(size) for size in current_shape)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise SystemExit(
                f"--current-shape requires three positive Z Y X values, got {shape}."
            )
        return shape, "--current-shape"

    source = "--current-zarr-path"
    resolved_path = current_zarr_path
    if resolved_path is None:
        resolved_path, source = _infer_current_zarr_path(project_folder)
    if resolved_path is None:
        raise SystemExit(
            "top-bottom-rebased needs the regenerated/current canvas shape. "
            "Pass --current-zarr-path (preferred), pass --current-shape Z Y X, "
            "or use --project-folder with an impp_settings.yaml whose "
            "job.output_path exists. The old --zarr-path shape cannot be reused "
            "because the bottom-view flip extent may have changed."
        )

    level_hint = (
        points_zarr_path.name
        if (points_zarr_path.expanduser() / ".zarray").exists()
        else None
    )
    shape, array_path = _spatial_shape_from_zarr(
        resolved_path,
        level_hint=level_hint,
    )
    return shape, f"{source}: {array_path}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open one or two Zarr arrays in napari with the point picker "
            "wired up for affine registration."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help=(
            "Path to a 5D Zarr array used as the source for any view whose "
            "own --fixed-zarr-path / --moving-zarr-path override is not "
            "supplied. No default -- you must pass at least one of these "
            "three options."
        ),
    )
    parser.add_argument(
        "--fixed-zarr-path",
        type=Path,
        default=None,
        help="Override the Zarr array used for the fixed view.",
    )
    parser.add_argument(
        "--moving-zarr-path",
        type=Path,
        default=None,
        help="Override the Zarr array used for the moving view.",
    )
    parser.add_argument(
        "--timepoint",
        type=int,
        default=0,
        help="Timepoint index to load.",
    )
    parser.add_argument(
        "--fixed-channel",
        type=int,
        default=0,
        help="Channel index used as the fixed/reference view.",
    )
    parser.add_argument(
        "--moving-channel",
        type=int,
        default=2,
        help="Channel index used as the moving view.",
    )
    parser.add_argument(
        "--project-folder",
        type=Path,
        default=None,
        help=(
            "Optional impp output/project folder. If it contains "
            "impp_settings.yaml or views_alignment.yaml with per-view "
            "transforms, the moving view transform is inferred and composed "
            "into the copy-paste YAML output. If no transform is found, raw "
            "data mode is assumed."
        ),
    )
    parser.add_argument(
        "--transform-yaml",
        type=Path,
        default=None,
        help=(
            "Optional transform YAML containing view_transforms. In view "
            "alignment mode this is used as the source of A_old when the "
            "project folder does not contain impp_settings.yaml."
        ),
    )
    parser.add_argument(
        "--fixed-view-id",
        type=int,
        default=None,
        help=(
            "Override the fixed view id used in transform inference. By "
            "default this is parsed from the channel label, then falls back "
            "to --fixed-channel."
        ),
    )
    parser.add_argument(
        "--moving-view-id",
        type=int,
        default=None,
        help=(
            "Override the moving view id used in transform inference. By "
            "default this is parsed from the channel label, then falls back "
            "to --moving-channel."
        ),
    )
    parser.add_argument(
        "--fixed-camera-id",
        type=int,
        default=None,
        help="Optional fixed/reference camera id for output labels.",
    )
    parser.add_argument(
        "--moving-camera-id",
        type=int,
        default=None,
        help=(
            "Optional moving camera id. Used for camera_alignment output and "
            "for per-camera scopes in views_alignment.yaml."
        ),
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("camera", "view"),
        default="camera",
        help=(
            "Select the affine estimator. 'camera' solves only the XY camera "
            "relationship with Z fixed. 'view' is for view-to-view alignment "
            "after deskew and permits Z translation plus in-plane geometry."
        ),
    )
    parser.add_argument(
        "--output-inverse",
        action="store_true",
        help=(
            "Print the inverse of the solved slot affine. Use this when the "
            "matrix estimated by the point picker has the opposite direction "
            "from the pipeline slot being updated."
        ),
    )
    parser.add_argument(
        "--invert-estimated-affine",
        action="store_true",
        help=(
            "Invert the point-picker estimated affine before solving the YAML "
            "slot. This changes the correction direction; unlike "
            "--output-inverse, it does not invert the emitted transform."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=(
            "moving-slot",
            "shared-space",
            "top-bottom-direct",
            "top-bottom-rebased",
        ),
        default="moving-slot",
        help=(
            "Choose the YAML output. 'moving-slot' updates only the moving "
            "view/camera slot. 'shared-space' emits fixed and moving "
            "view_transforms that move both views into a common halfway "
            "space (same pair). 'top-bottom-direct' aligns a moving pair "
            "onto the fixed pair (e.g. fixed view 0, moving view 2) when "
            "the loaded dataset was generated with the SAME transforms "
            "being updated: the picked correction is applied to both "
            "moving-pair views. 'top-bottom-rebased' is the headless "
            "variant for when the moving pair already has NEW pending "
            "transforms: point pairs are loaded from --point-pairs-json "
            "(picked on the OLD dataset described by "
            "--points-project-folder), rebased into the new frame, then "
            "solved like top-bottom-direct."
        ),
    )
    parser.add_argument(
        "--point-pairs-json",
        type=Path,
        default=None,
        help=(
            "Saved point-pairs JSON (as written by the point picker) used "
            "by --output-mode top-bottom-rebased. layer1 = fixed view, "
            "layer2 = moving view."
        ),
    )
    parser.add_argument(
        "--points-project-folder",
        type=Path,
        default=None,
        help=(
            "impp output folder of the dataset the point pairs were picked "
            "on (its impp_settings.yaml/views_alignment.yaml give the OLD "
            "transforms and offsets). Used by top-bottom-rebased."
        ),
    )
    parser.add_argument(
        "--points-transform-yaml",
        type=Path,
        default=None,
        help=(
            "Alternative to --points-project-folder: transform YAML with "
            "the view_transforms the picked dataset was generated with."
        ),
    )
    parser.add_argument(
        "--current-zarr-path",
        type=Path,
        default=None,
        help=(
            "Regenerated/current Zarr array (or OME-Zarr root) whose spatial "
            "shape defines the new pipeline flip extent. Used only by "
            "top-bottom-rebased. If omitted, the tool tries job.output_path "
            "from --project-folder."
        ),
    )
    parser.add_argument(
        "--current-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help=(
            "Explicit regenerated/current spatial shape for "
            "top-bottom-rebased, used when the current Zarr has not been "
            "generated yet. Mutually exclusive with --current-zarr-path."
        ),
    )
    parser.add_argument(
        "--save-rebased-points",
        type=Path,
        default=None,
        help=(
            "Optional path to write the rebased point pairs as a "
            "point-picker JSON (top-bottom-rebased only), so they can be "
            "reloaded in the GUI on the regenerated dataset."
        ),
    )
    parser.add_argument(
        "--shared-space-offset-mode",
        choices=("split-offset", "preserve-post"),
        default="split-offset",
        help=(
            "How shared-space output handles existing view-pair POST offsets. "
            "'split-offset' recenters the fixed/moving POST translations "
            "around their pair midpoint before solving, so an existing "
            "moving-only offset is distributed across both output transforms; "
            "the printed view-pair offset should then be set to zero. "
            "'preserve-post' keeps the previous asymmetric POST behavior."
        ),
    )
    parser.add_argument(
        "--show-full-pipeline-matrices",
        action="store_true",
        help=(
            "Also print diagnostic POST @ output @ PREV matrices. These are "
            "not copy-paste values for view_transforms."
        ),
    )
    parser.add_argument(
        "--allow-identity-old-transform",
        action="store_true",
        help=(
            "Allow view alignment mode to use identity for A_old when no old "
            "view transform is found. This is intended only for raw/identity "
            "slot debugging."
        ),
    )
    parser.add_argument(
        "--flip-fixed-y",
        action="store_true",
        help="Flip the fixed/reference view vertically along Y.",
    )
    parser.add_argument(
        "--flip-fixed-x",
        action="store_true",
        help="Flip the fixed/reference view horizontally along X.",
    )
    parser.add_argument(
        "--no-flip-moving-y",
        action="store_true",
        help=(
            "Disable the default moving Y flip used in camera mode. View mode "
            "does not flip moving Y by default."
        ),
    )
    parser.add_argument(
        "--flip-moving-y",
        action="store_true",
        help="Explicitly flip the moving view vertically along Y.",
    )
    parser.add_argument(
        "--flip-moving-x",
        action="store_true",
        help=(
            "Flip the moving view horizontally along X. Useful when aligning "
            "views from different cameras."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shared_path = (
        args.zarr_path.expanduser() if args.zarr_path is not None else None
    )
    fixed_zarr_path = (
        args.fixed_zarr_path.expanduser()
        if args.fixed_zarr_path is not None
        else shared_path
    )
    moving_zarr_path = (
        args.moving_zarr_path.expanduser()
        if args.moving_zarr_path is not None
        else shared_path
    )
    if fixed_zarr_path is None or moving_zarr_path is None:
        raise SystemExit(
            "No Zarr path provided. Pass --zarr-path for a shared source, or "
            "--fixed-zarr-path and --moving-zarr-path to load each view from "
            "its own array."
        )

    fixed_data = da.from_zarr(str(fixed_zarr_path))
    moving_data = (
        fixed_data
        if moving_zarr_path == fixed_zarr_path
        else da.from_zarr(str(moving_zarr_path))
    )

    if args.no_flip_moving_y and args.flip_moving_y:
        raise SystemExit(
            "Use only one of --no-flip-moving-y and --flip-moving-y."
        )
    if args.output_mode == "shared-space" and args.alignment_mode != "view":
        raise SystemExit(
            "--output-mode shared-space is supported only with "
            "--alignment-mode view."
        )
    top_bottom_mode = args.output_mode in (
        "top-bottom-direct",
        "top-bottom-rebased",
    )
    if top_bottom_mode and args.alignment_mode != "view":
        raise SystemExit(
            f"--output-mode {args.output_mode} is supported only with "
            "--alignment-mode view."
        )
    headless = args.output_mode == "top-bottom-rebased"
    if headless and args.point_pairs_json is None:
        raise SystemExit(
            "--output-mode top-bottom-rebased requires --point-pairs-json "
            "with the pairs picked on the old dataset."
        )
    if headless and (
        args.points_project_folder is None
        and args.points_transform_yaml is None
    ):
        raise SystemExit(
            "--output-mode top-bottom-rebased requires "
            "--points-project-folder (or --points-transform-yaml) "
            "describing the dataset the points were picked on."
        )

    default_moving_flip_y = args.alignment_mode == "camera"
    moving_flip_y = args.flip_moving_y or (
        default_moving_flip_y and not args.no_flip_moving_y
    )

    fixed_flips = ViewFlips(y=args.flip_fixed_y, x=args.flip_fixed_x)
    moving_flips = ViewFlips(
        y=moving_flip_y,
        x=args.flip_moving_x,
    )

    if headless:
        # Rebased mode never displays data: points come from the JSON in
        # original data coordinates. The loaded arrays describe the OLD point
        # frame; current pipeline slots need the independently resolved NEW
        # canvas shape because it defines the bottom-view Y-flip extent.
        fixed = None
        moving = None
        points_fixed_shape_zyx = tuple(int(s) for s in fixed_data.shape[2:])
        points_moving_shape_zyx = tuple(int(s) for s in moving_data.shape[2:])
        current_shape_zyx, current_shape_source = (
            _resolve_current_spatial_shape(
                current_shape=args.current_shape,
                current_zarr_path=args.current_zarr_path,
                project_folder=args.project_folder,
                points_zarr_path=fixed_zarr_path,
            )
        )
        fixed_shape_zyx = current_shape_zyx
        moving_shape_zyx = current_shape_zyx
    else:
        fixed = _slice_channel(
            fixed_data,
            timepoint=args.timepoint,
            channel=args.fixed_channel,
            flips=fixed_flips,
        )
        moving = _slice_channel(
            moving_data,
            timepoint=args.timepoint,
            channel=args.moving_channel,
            flips=moving_flips,
        )
        fixed_shape_zyx = tuple(fixed.shape)
        moving_shape_zyx = tuple(moving.shape)
        points_fixed_shape_zyx = fixed_shape_zyx
        points_moving_shape_zyx = moving_shape_zyx

    fixed_name = (
        f"View {args.fixed_channel}: "
        f"{_channel_label(fixed_zarr_path, args.fixed_channel)}"
    )
    moving_name = (
        f"View {args.moving_channel}: "
        f"{_channel_label(moving_zarr_path, args.moving_channel)}"
    )
    fixed_flip_tags = fixed_flips.tags()
    moving_flip_tags = moving_flips.tags()
    if fixed_flip_tags:
        fixed_name += f" ({', '.join(fixed_flip_tags)})"
    if moving_flip_tags:
        moving_name += f" ({', '.join(moving_flip_tags)})"
    moving_name += " (moving view)"

    print(f"Fixed zarr:  {fixed_zarr_path}")
    print(f"Moving zarr: {moving_zarr_path}")
    print(f"Fixed shape (T, C, Z, Y, X):  {fixed_data.shape}")
    print(f"Moving shape (T, C, Z, Y, X): {moving_data.shape}")
    print(
        f"Loaded timepoint {args.timepoint}, "
        f"fixed channel {args.fixed_channel}, "
        f"moving channel {args.moving_channel}"
    )
    if fixed_flip_tags:
        print(f"Fixed view flips: {', '.join(fixed_flip_tags)}")
    if moving_flip_tags:
        print(f"Moving view flips: {', '.join(moving_flip_tags)}")

    preferred_view_ids: set[int] | None = None
    if args.alignment_mode == "view":
        project_view_transforms, _ = _project_view_transforms(
            args.project_folder,
            args.transform_yaml,
        )
        if project_view_transforms:
            preferred_view_ids = set(project_view_transforms)

    fixed_view_id, fixed_view_source = _infer_view_id(
        args.fixed_view_id,
        fixed_zarr_path,
        args.fixed_channel,
        preferred_view_ids=preferred_view_ids,
    )
    moving_view_id, moving_view_source = _infer_view_id(
        args.moving_view_id,
        moving_zarr_path,
        args.moving_channel,
        preferred_view_ids=preferred_view_ids,
    )
    fixed_camera_id, fixed_camera_source = _infer_camera_id(
        args.fixed_camera_id,
        fixed_zarr_path,
        args.fixed_channel,
        args.project_folder,
    )
    moving_camera_id, moving_camera_source = _infer_camera_id(
        args.moving_camera_id,
        moving_zarr_path,
        args.moving_channel,
        args.project_folder,
    )
    if args.alignment_mode == "view":
        if fixed_camera_id is None and moving_camera_id is not None:
            fixed_camera_id = moving_camera_id
            fixed_camera_source = f"same as moving ({moving_camera_source})"
        if moving_camera_id is None and fixed_camera_id is not None:
            moving_camera_id = fixed_camera_id
            moving_camera_source = f"same as fixed ({fixed_camera_source})"

    moving_pair_view_id = None
    if top_bottom_mode:
        moving_pair_view_id = _top_bottom_partner_or_raise(
            fixed_view_id,
            moving_view_id,
        )

    fixed_slot = None
    if args.output_mode == "shared-space" or top_bottom_mode:
        fixed_slot = _infer_pipeline_slot(
            alignment_mode=args.alignment_mode,
            project_folder=args.project_folder,
            transform_yaml=args.transform_yaml,
            view_id=fixed_view_id,
            camera_id=fixed_camera_id,
            view_shape_zyx=fixed_shape_zyx,
            allow_identity_old_transform=args.allow_identity_old_transform,
        )
    moving_slot = _infer_pipeline_slot(
        alignment_mode=args.alignment_mode,
        project_folder=args.project_folder,
        transform_yaml=args.transform_yaml,
        view_id=moving_view_id,
        camera_id=moving_camera_id,
        view_shape_zyx=moving_shape_zyx,
        allow_identity_old_transform=args.allow_identity_old_transform,
    )
    moving_pair_slot = None
    if top_bottom_mode:
        moving_pair_slot = _infer_pipeline_slot(
            alignment_mode=args.alignment_mode,
            project_folder=args.project_folder,
            transform_yaml=args.transform_yaml,
            view_id=moving_pair_view_id,
            camera_id=moving_camera_id,
            view_shape_zyx=moving_shape_zyx,
            allow_identity_old_transform=args.allow_identity_old_transform,
        )
    points_fixed_slot = None
    points_moving_slot = None
    if headless:
        # Slots describing the dataset the points were picked on use the old
        # loaded shapes. Current slots above use the separately resolved new
        # shapes, so a changed bottom-view flip origin is rebased exactly.
        points_fixed_slot = _infer_pipeline_slot(
            alignment_mode=args.alignment_mode,
            project_folder=args.points_project_folder,
            transform_yaml=args.points_transform_yaml,
            view_id=fixed_view_id,
            camera_id=fixed_camera_id,
            view_shape_zyx=points_fixed_shape_zyx,
            allow_identity_old_transform=args.allow_identity_old_transform,
        )
        points_moving_slot = _infer_pipeline_slot(
            alignment_mode=args.alignment_mode,
            project_folder=args.points_project_folder,
            transform_yaml=args.points_transform_yaml,
            view_id=moving_view_id,
            camera_id=moving_camera_id,
            view_shape_zyx=points_moving_shape_zyx,
            allow_identity_old_transform=args.allow_identity_old_transform,
        )
    transform_inference = TransformInference(
        fixed_camera_id=fixed_camera_id,
        moving_camera_id=moving_camera_id,
        fixed_view_id=fixed_view_id,
        moving_view_id=moving_view_id,
        fixed_slot=fixed_slot,
        moving_slot=moving_slot,
        output_mode=args.output_mode,
        invert_estimated_affine=args.invert_estimated_affine,
        output_inverse=args.output_inverse,
        show_full_pipeline_matrices=args.show_full_pipeline_matrices,
        shared_space_offset_mode=args.shared_space_offset_mode,
        fixed_display_affine=(
            _identity4()
            if headless
            else _display_affine_for_flips(fixed_shape_zyx, fixed_flips)
        ),
        moving_display_affine=(
            _identity4()
            if headless
            else _display_affine_for_flips(moving_shape_zyx, moving_flips)
        ),
        fixed_view_source=fixed_view_source,
        moving_view_source=moving_view_source,
        moving_pair_slot=moving_pair_slot,
        moving_pair_view_id=moving_pair_view_id,
        points_fixed_slot=points_fixed_slot,
        points_moving_slot=points_moving_slot,
    )

    print(
        f"Fixed view id: {fixed_view_id} "
        f"({transform_inference.fixed_view_source})"
    )
    print(
        f"Moving view id: {moving_view_id} "
        f"({transform_inference.moving_view_source})"
    )
    if fixed_camera_id is not None:
        print(f"Fixed camera id: {fixed_camera_id} ({fixed_camera_source})")
    if moving_camera_id is not None:
        print(f"Moving camera id: {moving_camera_id} ({moving_camera_source})")
    print(f"Output mode: {args.output_mode}")
    if args.output_mode == "shared-space":
        print(f"Shared-space offset mode: {args.shared_space_offset_mode}")
    estimated_direction = _estimated_direction(args.invert_estimated_affine)
    direction_note = (
        "inverted before solve"
        if args.invert_estimated_affine
        else "as estimated"
    )
    print(
        "Estimated affine direction: "
        f"{estimated_direction} ({direction_note})"
    )
    if fixed_slot is not None:
        print(f"Fixed pipeline slot target: {fixed_slot.output_target}")
        print(f"Fixed A_old source: {fixed_slot.old_source}")
        print(f"Fixed POST source: {fixed_slot.post_source}")
        print(f"Fixed PREV source: {fixed_slot.prev_source}")
    print(f"Moving pipeline slot target: {moving_slot.output_target}")
    print(f"Moving A_old source: {moving_slot.old_source}")
    print(f"Moving POST source: {moving_slot.post_source}")
    print(f"Moving PREV source: {moving_slot.prev_source}")
    if moving_pair_slot is not None:
        print(f"Partner view id: {moving_pair_view_id}")
        print(f"Partner A_old source: {moving_pair_slot.old_source}")
        print(f"Partner POST source: {moving_pair_slot.post_source}")
    if points_moving_slot is not None:
        print(f"Points frame shape: {points_moving_shape_zyx}")
        print(
            f"Current frame shape: {current_shape_zyx} "
            f"({current_shape_source})"
        )
        print(f"Points A_old source: {points_moving_slot.old_source}")
        print(f"Points POST source: {points_moving_slot.post_source}")

    if headless:
        affine_estimator, estimator_label = _build_affine_estimator(
            args.alignment_mode
        )
        print(f"Affine estimator: {estimator_label}")
        _run_top_bottom_rebased(args, transform_inference, affine_estimator)
        return

    viewer = napari.Viewer(title="Zarr view registration")
    viewer.add_image(
        fixed,
        name=fixed_name,
        colormap="green",
        blending="additive",
        contrast_limits=(0, 65535),
        metadata={
            "source_zarr": str(fixed_zarr_path),
            "timepoint": args.timepoint,
            "channel": args.fixed_channel,
            "y_flipped": fixed_flips.y,
            "x_flipped": fixed_flips.x,
        },
    )
    viewer.add_image(
        moving,
        name=moving_name,
        colormap="magenta",
        blending="additive",
        contrast_limits=(0, 65535),
        metadata={
            "source_zarr": str(moving_zarr_path),
            "timepoint": args.timepoint,
            "channel": args.moving_channel,
            "y_flipped": moving_flips.y,
            "x_flipped": moving_flips.x,
        },
    )

    viewer.dims.current_step = tuple(int(size // 2) for size in fixed.shape)

    affine_estimator, estimator_label = _build_affine_estimator(
        args.alignment_mode
    )
    print(f"Affine estimator: {estimator_label}")

    manager = show_point_picker(
        viewer,
        layer1_name=fixed_name,
        layer2_name=moving_name,
        affine_estimator=affine_estimator,
    )
    manager.point_picker_widget.affine_applied.connect(
        lambda _affine: _print_yaml_ready_affine(
            manager,
            transform_inference,
        )
    )
    napari.run()


if __name__ == "__main__":
    main()
