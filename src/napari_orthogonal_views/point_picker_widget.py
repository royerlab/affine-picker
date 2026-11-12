import datetime
import getpass
import html
import itertools
import json
import math
import os
import re
import traceback
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import affiners
import numpy as np
from napari.components import ViewerModel
from napari.utils.action_manager import action_manager
from napari.utils.notifications import show_error, show_info, show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_orthogonal_views.fit_diagnostics import (
    AXIS_NAMES_3D,
    FitReport,
    apply_affine,
    evaluate_fit,
)

# Layer metadata key holding the world position of full-resolution voxel
# (0, 0, 0). Set it (with ``layer.scale`` equal to the downsampling factor)
# when the layer shows a downsampled pyramid level, so picks, residuals and
# the estimated affine are all in full-resolution voxels. See
# ``downsampled_layer_kwargs``.
FULL_RES_ORIGIN_KEY = "full_res_origin"

PICK_FILE_VERSION = 2
DEFAULT_AUTOSAVE_DIR = Path.home() / ".napari-point-picker" / "autosave"
# Literal /tmp, not $TMPDIR: a Slurm job's TMPDIR is deleted when it ends.
DEFAULT_BACKUP_DIR = Path("/tmp") / f"point-picker-{getpass.getuser()}"
UNDO_LIMIT = 200

# (napari shortcut, display text, PointPickerWidget method, description).
# A method of None marks a key bound elsewhere (T, in ortho_view_manager).
KEYBINDINGS = (
    ("T", "T", None, "crosshair to mouse"),
    ("Q", "Q", "record_fixed", "set fixed point"),
    ("W", "W", "record_moving", "set moving point"),
    ("Shift-Q", "Shift+Q", "go_to_fixed", "go to fixed point"),
    ("Shift-W", "Shift+W", "go_to_moving", "go to moving point"),
    ("N", "N", "add_pair", "new pair"),
    ("[", "[", "select_previous_pair", "previous pair"),
    ("]", "]", "select_next_pair", "next pair"),
    ("U", "U", "toggle_active_use", "use or skip pair in fit"),
    ("Control-Z", "Ctrl+Z", "undo", "undo"),
)

_ACTIVE_PICKER = None  # weakref to the most recently created picker
_ACTIONS_REGISTERED = False


def _active_picker():
    return _ACTIVE_PICKER() if _ACTIVE_PICKER is not None else None


def _make_command(method_name: str):
    def command(viewer_model: ViewerModel):
        picker = _active_picker()
        if picker is None:
            return
        # Keys pressed in a 3D preview window must not edit the picks.
        if any(viewer_model is p for p in picker._preview_viewers):
            return
        getattr(picker, method_name)()

    command.__name__ = f"point_picker_{method_name}"
    return command


def _register_actions() -> None:
    """Bind the picker hotkeys on every viewer model (main and ortho)."""
    global _ACTIONS_REGISTERED
    if _ACTIONS_REGISTERED:
        return
    for shortcut, _, method, description in KEYBINDINGS:
        if method is None:
            continue
        name = f"napari:point_picker_{method}"
        action_manager.register_action(
            name=name,
            command=_make_command(method),
            description=f"Point picker: {description}",
            keymapprovider=ViewerModel,
        )
        action_manager.bind_shortcut(name, shortcut)
    _ACTIONS_REGISTERED = True


def downsampled_layer_kwargs(
    factor,
    full_res_translate=None,
    ndim: int = 3,
) -> dict:
    """``add_image`` keyword arguments for a downsampled pyramid level.

    With these, napari world coordinates are full-resolution voxels: the
    layer is drawn at ``scale = factor`` and shifted by half a coarse voxel
    so each coarse voxel sits over the block of full-resolution voxels it
    summarises (coarse voxel ``i`` covers full-resolution indices
    ``factor * i`` to ``factor * i + factor - 1``). The point picker then
    records picks, residuals and the estimated affine in full-resolution
    voxels, whatever level is on screen.

    Parameters
    ----------
    factor : float or sequence of float
        Downsampling factor per axis (for example 4 for pyramid level 2 of a
        2x pyramid).
    full_res_translate : sequence of float, optional
        Translate the full-resolution layer would have, in full-resolution
        voxels. Defaults to zero.
    ndim : int
        Number of spatial axes.
    """
    factor = np.broadcast_to(np.asarray(factor, dtype=float), (ndim,))
    origin = (
        np.zeros(ndim)
        if full_res_translate is None
        else np.asarray(full_res_translate, dtype=float)
    )
    return {
        "scale": tuple(factor.tolist()),
        "translate": tuple((origin + (factor - 1) / 2).tolist()),
        "metadata": {FULL_RES_ORIGIN_KEY: origin.tolist()},
    }


def downsample_for_preview(
    data, scale, translate, max_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrink ``data`` so its longest axis is at most ``max_size`` voxels.

    Takes the maximum of each ``step``-sized block rather than striding, so
    a bead a voxel or two wide is not dropped (it would be wherever it fell
    between the kept voxels). Returns the result (computed, if ``data`` is
    lazy) with the scale and translate that put each block at its centre.
    """
    step = max(1, math.ceil(max(data.shape) / max_size))
    scale = np.asarray(scale, dtype=float)
    translate = np.asarray(translate, dtype=float)
    if step == 1:
        return _finite(np.asarray(data)), scale, translate
    if hasattr(data, "map_blocks"):  # dask: reduce chunk by chunk
        import dask.array as da

        padded = da.pad(
            data,
            [(0, (-n) % step) for n in data.shape],
            mode="constant",
        )
        blocks = da.coarsen(
            np.max, padded, dict.fromkeys(range(data.ndim), step)
        ).compute()
    else:
        array = np.asarray(data)
        padded = np.pad(
            array,
            [(0, (-n) % step) for n in array.shape],
            mode="constant",
        )
        shape = []
        for n in padded.shape:
            shape += [n // step, step]
        blocks = padded.reshape(shape).max(
            axis=tuple(range(1, 2 * array.ndim, 2))
        )
    return (
        _finite(np.asarray(blocks)),
        scale * step,
        translate + scale * (step - 1) / 2,
    )


def _finite(array: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 and infinities with the dtype's extremes.

    A stray NaN or inf (float16 stores have a few) spoils a maximum
    intensity projection along every ray through it.
    """
    if np.issubdtype(array.dtype, np.floating) and not np.all(
        np.isfinite(array)
    ):
        return np.nan_to_num(array, copy=True)
    return array


def _json_safe(value, depth: int = 0):
    """Return a JSON-serialisable copy of layer metadata, dropping the rest."""
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            safe = _json_safe(item, depth + 1)
            if safe is not None or item is None:
                out[str(key)] = safe
        return out
    return None


def _describe_estimator(estimator) -> str:
    func = getattr(estimator, "func", estimator)
    name = getattr(func, "__name__", repr(func))
    keywords = getattr(estimator, "keywords", None)
    if keywords:
        args = ", ".join(f"{k}={v!r}" for k, v in keywords.items())
        return f"{name}({args})"
    return name


def _fmt_coords(coords) -> str:
    return ", ".join(f"{c:.1f}" for c in coords)


@dataclass
class PointPair:
    """Store a matched pair of coordinates for registration."""

    pair_id: int
    layer1_coords: tuple | None = None
    layer2_coords: tuple | None = None
    enabled: bool = True


class ShowUpdateWidget(QWidget):
    """Table cell showing one coordinate, with Show and Update buttons."""

    show_clicked = Signal()
    update_clicked = Signal()

    def __init__(self):
        super().__init__()

        self.coords_label = QLabel("not set")
        self.coords_label.setAlignment(Qt.AlignCenter)
        self.show_button = QPushButton("Show")
        self.update_button = QPushButton("Update")

        self.show_button.clicked.connect(self.show_clicked.emit)
        self.update_button.clicked.connect(self.update_clicked.emit)

        buttons = QHBoxLayout()
        buttons.addWidget(self.show_button)
        buttons.addWidget(self.update_button)
        buttons.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout()
        layout.addWidget(self.coords_label)
        layout.addLayout(buttons)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        self.setLayout(layout)

    def set_coords(self, coords) -> None:
        if coords is None:
            self.coords_label.setText("not set")
            self.show_button.setEnabled(False)
        else:
            self.coords_label.setText(_fmt_coords(coords))
            self.show_button.setEnabled(True)


def _validate_point_arrays(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce point arrays for affine estimation."""
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)

    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Point arrays must be 2-dimensional (N, ndim).")
    if source_points.shape != target_points.shape:
        raise ValueError(
            "Source and target point arrays must have the same shape."
        )

    n, ndim = source_points.shape
    if n < ndim + 1:
        raise ValueError(
            f"Need at least {ndim + 1} point pairs for {ndim}D affine estimation, "
            f"got {n}."
        )
    return source_points, target_points


def estimate_affine_from_points(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Estimate a full affine transform (with scaling) from matched point pairs.

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix.

    Raises
    ------
    ValueError
        If fewer than ndim+1 point pairs are provided.
    """
    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    n, ndim = source_points.shape

    # Add homogeneous coordinate
    src_homo = np.hstack([source_points, np.ones((n, 1))])

    # Solve least squares: tgt = src_homo @ T.T
    T, _, _, _ = np.linalg.lstsq(src_homo, target_points, rcond=None)

    # Build full homogeneous matrix
    affine_matrix = np.eye(ndim + 1)
    affine_matrix[:ndim, :] = T.T

    return affine_matrix


def estimate_affine_from_points_no_scale(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Estimate an affine transform without scaling (diagonal of linear part fixed to 1).

    Solves constrained least squares column-by-column. For each target dimension j,
    the coefficient T[j,j] (the scale along axis j) is fixed to 1 and the remaining
    coefficients (off-diagonal shear/rotation + translation) are optimized.

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix with ones on the linear diagonal.

    Raises
    ------
    ValueError
        If fewer than ndim+1 point pairs are provided.
    """
    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    n, ndim = source_points.shape

    src_homo = np.hstack([source_points, np.ones((n, 1))])
    affine_matrix = np.eye(ndim + 1)

    for j in range(ndim):
        # Move the known diagonal contribution (1 * source_j) to the RHS
        rhs = target_points[:, j] - source_points[:, j]

        # Solve for all coefficients except the diagonal one
        other_cols = [i for i in range(ndim + 1) if i != j]
        lhs = src_homo[:, other_cols]

        coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

        # affine_matrix[j, j] is already 1 from np.eye
        for k, col_idx in enumerate(other_cols):
            if col_idx < ndim:
                affine_matrix[j, col_idx] = coeffs[k]
            else:
                affine_matrix[j, ndim] = coeffs[k]  # translation

    return affine_matrix


def estimate_affine_from_points_masked(
    source_points: np.ndarray,
    target_points: np.ndarray,
    mask: np.ndarray,
    fixed_values: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate an affine with a user-specified subset of parameters free.

    Generalizes :func:`estimate_affine_from_points_no_scale`: instead of
    hard-coding which entries of the linear+translation block are fixed,
    the caller passes a boolean mask the same shape as that block.

    For each target dimension ``j``, the entries of row ``j`` of the affine
    flagged as free in ``mask`` are optimized via least squares; the rest
    are held to ``fixed_values[j]`` (default: identity — ones on the
    diagonal, zeros elsewhere).

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.
    mask : array_like of bool
        (ndim, ndim+1) mask. ``True`` (or any truthy value) marks an entry
        as free to optimize; ``False`` holds it at ``fixed_values``.
    fixed_values : np.ndarray, optional
        (ndim, ndim+1) values used for entries where ``mask`` is False.
        Defaults to ``np.eye(ndim, ndim+1)`` so non-optimized entries take
        identity values.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix.

    Raises
    ------
    ValueError
        If point arrays are inconsistent, ``mask``/``fixed_values`` shapes
        don't match ``(ndim, ndim+1)``, or any row of ``mask`` has more
        free parameters than data points.
    """
    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    n, ndim = source_points.shape

    mask = np.asarray(mask).astype(bool)
    expected_shape = (ndim, ndim + 1)
    if mask.shape != expected_shape:
        raise ValueError(
            f"mask must have shape {expected_shape}, got {mask.shape}."
        )

    if fixed_values is None:
        fixed_values = np.eye(ndim, ndim + 1)
    else:
        fixed_values = np.asarray(fixed_values, dtype=float)
        if fixed_values.shape != expected_shape:
            raise ValueError(
                f"fixed_values must have shape {expected_shape}, "
                f"got {fixed_values.shape}."
            )

    src_homo = np.hstack([source_points, np.ones((n, 1))])
    affine_matrix = np.eye(ndim + 1)
    affine_matrix[:ndim, :] = fixed_values

    for j in range(ndim):
        free_idx = np.flatnonzero(mask[j])
        fixed_idx = np.flatnonzero(~mask[j])

        if free_idx.size == 0:
            continue
        if free_idx.size > n:
            raise ValueError(
                f"Row {j} has {free_idx.size} free parameters but only {n} "
                "point pairs are available."
            )

        # Subtract the contribution of the held-fixed entries from the RHS,
        # then solve least squares for the free ones.
        rhs = (
            target_points[:, j]
            - src_homo[:, fixed_idx] @ fixed_values[j, fixed_idx]
        )
        lhs = src_homo[:, free_idx]
        coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        affine_matrix[j, free_idx] = coeffs

    return affine_matrix


def _decompose_linear_3d(
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a 3x3 linear map L as ``R · diag(s) · H``.

    R is a rotation (so(3), returned as an axis-angle 3-vector), ``s`` are
    the three scales (positive when ``det(L) > 0``), and H is unit
    upper-triangular shear, returned as ``(h01, h02, h12)``.

    Used to seed :func:`estimate_affine_from_points_components` from the
    unconstrained least-squares estimate.
    """
    from scipy.spatial.transform import Rotation

    Q, K = np.linalg.qr(L)
    # Force positive diagonals on K by absorbing signs into Q
    signs = np.sign(np.diag(K))
    signs[signs == 0] = 1
    K = signs[:, None] * K
    Q = Q * signs[None, :]
    # Resolve reflections so Q is a proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
        K[-1, :] *= -1

    scale = np.diag(K).copy()
    H = K / scale[:, None]
    shear = np.array([H[0, 1], H[0, 2], H[1, 2]])
    rotvec = Rotation.from_matrix(Q).as_rotvec()
    return rotvec, scale, shear


def _compose_affine_3d(
    rotvec: np.ndarray,
    scale: np.ndarray,
    shear: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`_decompose_linear_3d`; builds the 4x4 affine."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(rotvec).as_matrix()
    S = np.diag(scale)
    H = np.array(
        [
            [1.0, shear[0], shear[1]],
            [0.0, 1.0, shear[2]],
            [0.0, 0.0, 1.0],
        ]
    )
    A = np.eye(4)
    A[:3, :3] = R @ S @ H
    A[:3, 3] = translation
    return A


def _normalize_component_mask(mask, name: str) -> np.ndarray:
    """Coerce a ``fix_*`` argument to a length-3 boolean array.

    A scalar bool broadcasts across all three axes; a length-3 sequence is
    converted as-is.
    """
    if isinstance(mask, (bool, np.bool_)):
        return np.array([bool(mask)] * 3, dtype=bool)
    arr = np.asarray(mask)
    if arr.shape != (3,):
        raise ValueError(
            f"{name} must be a bool or a length-3 sequence of bools, "
            f"got shape {arr.shape}."
        )
    return arr.astype(bool)


def estimate_affine_from_points_components(
    source_points: np.ndarray,
    target_points: np.ndarray,
    fix_scale=False,
    fix_shear=False,
    fix_rotation=False,
    fix_translation=False,
) -> np.ndarray:
    """Estimate a 3D affine with selected per-axis components held fixed.

    The affine is parameterized as

        A = T · R · S · H

    with 12 free parameters total:

    - rotation: so(3) axis-angle ``(rx, ry, rz)``
    - scale:    diagonal ``(sx, sy, sz)``
    - shear:    unit-upper-triangular ``(h01, h02, h12)``
    - translation: ``(tx, ty, tz)``

    Each ``fix_<component>`` argument controls which entries of that
    component are held at the identity value during optimization. It can be:

    - a single bool — broadcast across all three axes;
    - a length-3 sequence of bools — per-axis (``True`` = fixed).

    Example: ``fix_rotation=(False, True, True)`` keeps ``rx`` free and
    pins ``ry``/``rz`` to 0.

    Identity values used for held entries are: scale=1, shear=0,
    rotation=0 (axis-angle), translation=0. Free entries are initialized
    from the QR decomposition of the unconstrained linear LS estimate,
    then refined via :func:`scipy.optimize.least_squares` (LM).

    Only 3D point sets are supported.

    Returns
    -------
    np.ndarray
        4x4 homogeneous affine matrix.
    """
    from scipy.optimize import least_squares

    source_points, target_points = _validate_point_arrays(
        source_points, target_points
    )
    _, ndim = source_points.shape
    if ndim != 3:
        raise ValueError(
            "estimate_affine_from_points_components supports only 3D points; "
            f"got ndim={ndim}."
        )

    fix_rot = _normalize_component_mask(fix_rotation, "fix_rotation")
    fix_sca = _normalize_component_mask(fix_scale, "fix_scale")
    fix_she = _normalize_component_mask(fix_shear, "fix_shear")
    fix_tra = _normalize_component_mask(fix_translation, "fix_translation")
    free_mask = np.concatenate([~fix_rot, ~fix_sca, ~fix_she, ~fix_tra])

    id_full = np.concatenate(
        [np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3)]
    )

    # Seed free entries from an unconstrained LS fit. If everything is
    # fixed there's nothing to optimize — return identity directly.
    if not free_mask.any():
        return np.eye(4)

    A_init = estimate_affine_from_points(source_points, target_points)
    try:
        init_rotvec, init_scale, init_shear = _decompose_linear_3d(
            A_init[:3, :3]
        )
        init_trans = A_init[:3, 3]
    except np.linalg.LinAlgError:
        init_rotvec = np.zeros(3)
        init_scale = np.ones(3)
        init_shear = np.zeros(3)
        init_trans = np.zeros(3)

    init_full = np.concatenate(
        [init_rotvec, init_scale, init_shear, init_trans]
    )
    x0 = init_full[free_mask]

    def unpack(x: np.ndarray) -> tuple:
        full = id_full.copy()
        full[free_mask] = x
        return full[0:3], full[3:6], full[6:9], full[9:12]

    def residuals(x: np.ndarray) -> np.ndarray:
        rotvec, scale, shear, trans = unpack(x)
        A = _compose_affine_3d(rotvec, scale, shear, trans)
        pred = (A[:3, :3] @ source_points.T).T + A[:3, 3]
        return (pred - target_points).ravel()

    result = least_squares(residuals, x0, method="lm")
    rotvec, scale, shear, trans = unpack(result.x)
    return _compose_affine_3d(rotvec, scale, shear, trans)


COL_PAIR, COL_USE, COL_FIXED, COL_MOVING, COL_RESID, COL_LOO, COL_DELETE = (
    range(7)
)
ORDER_PAIR, ORDER_RESIDUAL, ORDER_LOO = range(3)
FLAG_COLOUR = QColor(192, 57, 43, 120)
EXCLUDED_COLOUR = QColor(128, 128, 128)


class PointPickerWidget(QWidget):
    """Widget for picking matched point pairs across two image layers.

    The fit is re-estimated after every edit and its diagnostics (per-pair
    and leave-one-out residuals, stability, implausibility warnings) are
    shown without resampling the moving image; **Apply Estimated Affine**
    resamples it.

    Parameters
    ----------
    viewer
        The napari viewer holding both layers.
    layer1_name, layer2_name
        Names of the fixed (reference) and moving layers.
    affine_estimator
        ``(source, target) -> affine`` called with moving points as source.
    preview_data_provider
        Optional ``(layer_name, max_size) -> (array, scale, translate)``
        returning a low-resolution copy of a layer's source data for the 3D
        preview, in the same world frame as the layer (for example a coarser
        pyramid level). Return ``None`` to fall back to striding the layer.
    autosave_dir
        Directory for the autosave file written after every edit. ``None``
        disables autosave.
    backup_dir
        Directory for a second copy written after every edit (default
        ``/tmp/point-picker-<user>``), so picks survive a failing Save
        dialog or autosave location. ``None`` disables it.
    """

    affine_applied = Signal(np.ndarray)
    fit_updated = Signal(object)

    def __init__(
        self,
        viewer: Viewer,
        layer1_name: str = "Image 1",
        layer2_name: str = "Image 2",
        affine_estimator: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        preview_data_provider: Callable | None = None,
        autosave_dir: str | Path | None = DEFAULT_AUTOSAVE_DIR,
        backup_dir: str | Path | None = DEFAULT_BACKUP_DIR,
    ):
        super().__init__()

        self.viewer = viewer
        self.layer1_name = layer1_name
        self.layer2_name = layer2_name
        self.affine_estimator = affine_estimator or estimate_affine_from_points
        self.preview_data_provider = preview_data_provider
        self.autosave_dir = (
            Path(autosave_dir).expanduser()
            if autosave_dir is not None
            else None
        )
        self.autosave_path: Path | None = None
        self.backup_dir = (
            Path(backup_dir).expanduser() if backup_dir is not None else None
        )
        self.backup_path: Path | None = None
        self._save_failures: set[Path] = set()

        # Ordered by creation; the table can show another order.
        self.point_pairs: dict[int, PointPair] = {}
        self._next_pair_id = 1
        self._active_pair_id: int | None = None
        self._undo_stack: list[tuple] = []
        self.fit_report: FitReport | None = None

        self.transform_snapshot: dict | None = None
        self._applied_affine: np.ndarray | None = None
        self._original_translates: dict[str, np.ndarray] = {}
        self._centre_offsets: dict[str, np.ndarray] = {}
        self._translates_captured = False
        self._nudge_offset = np.zeros(3)
        self._affine_base_translate: np.ndarray | None = None
        # True once "Reset (keep view)" has re-baselined the moving layer,
        # after which its source data no longer matches what is displayed.
        self._rebased = False

        self._preview_viewers: list = []
        self._preview_workers: list = []
        self.preview_show_window = True

        self._building_table = False
        self._overlay_active = False
        self._row_cells: dict[int, tuple] = {}

        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(0)
        self._fit_timer.timeout.connect(self.refresh_fit)

        self._build_ui()
        _register_actions()
        global _ACTIVE_PICKER
        _ACTIVE_PICKER = weakref.ref(self)
        self.refresh_fit()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # Hotkeys stay above the scroll area so they are always visible.
        self.keys_label = QLabel(self._keys_html())
        self.keys_label.setTextFormat(Qt.RichText)
        self.keys_label.setWordWrap(True)
        self.keys_label.setFrameShape(QFrame.StyledPanel)
        self.keys_label.setToolTip(
            "Hotkeys work while the mouse is over the main view or an "
            "orthogonal view. Q and W record the crosshair position (move "
            "it with T first) into the active (selected) pair."
        )
        self.canvas_keys_checkbox = QCheckBox(
            "Show hotkeys and the active pair on the canvas"
        )
        self.canvas_keys_checkbox.setChecked(True)
        self.canvas_keys_checkbox.toggled.connect(
            lambda _checked: self._update_canvas_overlay()
        )

        self.layers_label = QLabel()
        self.layers_label.setWordWrap(True)
        self.layers_label.setTextFormat(Qt.RichText)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["#", "Use", "Fixed", "Moving", "Resid", "LOO", ""]
        )
        header_tips = {
            COL_PAIR: "Pair number, as used in all printed output.",
            COL_USE: "Use this pair in the fit (U toggles the active pair).",
            COL_FIXED: self.layer1_name,
            COL_MOVING: self.layer2_name,
            COL_RESID: (
                "Residual: distance between the fixed point and the moving "
                "point mapped through the current fit."
            ),
            COL_LOO: (
                "Leave-one-out residual: the same distance under a fit made "
                "without this pair. A mismatched pair pulls the full fit "
                "towards itself, so its own residual can look small; this "
                "column exposes it. For a pair not used in the fit it equals "
                "the residual."
            ),
        }
        for column, tip in header_tips.items():
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.horizontalHeader().setStretchLastSection(False)
        for column, width in (
            (COL_PAIR, 36),
            (COL_USE, 36),
            (COL_FIXED, 170),
            (COL_MOVING, 170),
            (COL_RESID, 56),
            (COL_LOO, 56),
            (COL_DELETE, 30),
        ):
            self.table.setColumnWidth(column, width)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemChanged.connect(self._on_item_changed)

        self.add_button = QPushButton("Add pair (N)")
        self.add_button.clicked.connect(lambda: self.add_pair())
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(lambda: self.undo())
        self.order_combo = QComboBox()
        self.order_combo.addItems(
            ["Order by pair #", "Order by residual", "Order by LOO"]
        )
        self.order_combo.currentIndexChanged.connect(
            lambda _index: self._rebuild_table()
        )
        self.flag_spin = QDoubleSpinBox()
        self.flag_spin.setRange(0.0, 1e5)
        self.flag_spin.setDecimals(2)
        self.flag_spin.setSingleStep(0.5)
        self.flag_spin.setValue(2.0)
        self.flag_spin.setPrefix("Flag > ")
        self.flag_spin.setToolTip(
            "Highlight pairs whose residual or leave-one-out residual is "
            "above this value."
        )
        self.flag_spin.valueChanged.connect(lambda _v: self._refresh_rows())
        pair_row = QHBoxLayout()
        pair_row.addWidget(self.add_button)
        pair_row.addWidget(self.undo_button)
        pair_row.addWidget(self.order_combo)
        pair_row.addWidget(self.flag_spin)

        self.fit_label = QLabel()
        self.fit_label.setWordWrap(True)
        self.fit_label.setTextFormat(Qt.RichText)
        self.fit_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fit_group = QGroupBox("Fit")
        fit_layout = QVBoxLayout()
        fit_layout.addWidget(self.fit_label)
        fit_group.setLayout(fit_layout)

        self.save_points_button = QPushButton("Save Points")
        self.save_points_button.clicked.connect(self._save_points_dialog)
        self.load_points_button = QPushButton("Load Points")
        self.load_points_button.clicked.connect(self._load_points_dialog)
        save_load_layout = QHBoxLayout()
        save_load_layout.addWidget(self.save_points_button)
        save_load_layout.addWidget(self.load_points_button)
        self.autosave_label = QLabel(
            f"Picks are autosaved to {self.autosave_dir} after every edit."
            if self.autosave_dir is not None
            else "Autosave is off."
        )
        self.autosave_label.setWordWrap(True)
        self.autosave_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        preview_group = QGroupBox("3D preview (opens a new window)")
        self.preview_size_spin = QSpinBox()
        self.preview_size_spin.setRange(8, 4096)
        self.preview_size_spin.setValue(384)
        self.preview_size_spin.setSuffix(" voxels")
        self.preview_size_spin.setPrefix("Longest axis ")
        self.preview_size_spin.setToolTip(
            "The preview is downsampled until its longest axis is at most "
            "this many voxels."
        )
        self.preview_raw_button = QPushButton("Raw data")
        self.preview_raw_button.setToolTip(
            "Both images at their original positions (no estimate, no "
            "nudge), with the picks."
        )
        self.preview_raw_button.clicked.connect(
            lambda: self.open_3d_preview(with_estimate=False)
        )
        self.preview_estimate_button = QPushButton("With current estimate")
        self.preview_estimate_button.setToolTip(
            "The moving image mapped through the current fit (applied or "
            "not), with the picks."
        )
        self.preview_estimate_button.clicked.connect(
            lambda: self.open_3d_preview(with_estimate=True)
        )
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.preview_size_spin)
        preview_layout.addWidget(self.preview_raw_button)
        preview_layout.addWidget(self.preview_estimate_button)
        preview_group.setLayout(preview_layout)

        nudge_group = QGroupBox("Nudge Offset")
        nudge_layout = QVBoxLayout()
        nudge_spin_layout = QHBoxLayout()
        self._nudge_spins = []
        for axis_label in ("Z", "Y", "X"):
            label = QLabel(axis_label)
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._apply_nudge)
            nudge_spin_layout.addWidget(label)
            nudge_spin_layout.addWidget(spin)
            self._nudge_spins.append(spin)
        nudge_layout.addLayout(nudge_spin_layout)
        self._reset_nudge_button = QPushButton("Reset Nudge")
        self._reset_nudge_button.clicked.connect(self._reset_nudge)
        nudge_layout.addWidget(self._reset_nudge_button)
        nudge_group.setLayout(nudge_layout)

        self.apply_button = QPushButton("Apply Estimated Affine")
        self.apply_button.clicked.connect(self._apply_affine)
        self.reset_button = QPushButton("Reset Transform")
        self.reset_button.clicked.connect(self._reset_transform)
        self.restart_button = QPushButton("Reset (keep view)")
        self.restart_button.setToolTip(
            "Clear point pairs and transform state but leave the moving "
            "layer's currently displayed image in place. Use this to pick "
            "a fresh set of correspondences on top of the partial alignment "
            "for iterative refinement."
        )
        self.restart_button.clicked.connect(self._restart_from_current)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.restart_button)

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.layers_label)
        content_layout.addWidget(self.table)
        content_layout.addLayout(pair_row)
        content_layout.addWidget(fit_group)
        content_layout.addLayout(save_load_layout)
        content_layout.addWidget(self.autosave_label)
        content_layout.addWidget(preview_group)
        content_layout.addWidget(nudge_group)
        content_layout.addLayout(button_layout)
        content.setLayout(content_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout()
        layout.addWidget(self.keys_label)
        layout.addWidget(self.canvas_keys_checkbox)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self._update_layers_label()

    @staticmethod
    def _keys_html() -> str:
        items = [
            f"<b>{html.escape(display)}</b> {html.escape(description)}"
            for _, display, _, description in KEYBINDINGS
        ]
        return "Hotkeys: " + " &nbsp;·&nbsp; ".join(items)

    def _units_phrase(self) -> str:
        if self._uses_full_res_units():
            return "full-resolution voxels"
        return "world units"

    def _update_layers_label(self) -> None:
        self.layers_label.setText(
            f"<b>Fixed:</b> {html.escape(self.layer1_name)}<br>"
            f"<b>Moving:</b> {html.escape(self.layer2_name)}<br>"
            f"Coordinates and residuals are in {self._units_phrase()}."
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._overlay_active:
            self.set_canvas_overlay_active(True)

    def set_canvas_overlay_active(self, active: bool) -> None:
        """Show or hide the hotkey overlay on the main canvas."""
        self._overlay_active = active
        self._update_canvas_overlay()

    def _update_canvas_overlay(self) -> None:
        overlay = self.viewer.text_overlay
        if not (self._overlay_active and self.canvas_keys_checkbox.isChecked()):
            if getattr(self, "_overlay_shown", False):
                overlay.visible = False
                self._overlay_shown = False
            return
        overlay.text = self._overlay_text()
        overlay.font_size = 9
        overlay.visible = True
        self._overlay_shown = True

    def _overlay_text(self) -> str:
        pair = self.point_pairs.get(self._active_pair_id)
        if pair is None:
            status = "Point picker: no active pair (N for a new one)"
        else:
            fixed = "set" if pair.layer1_coords is not None else "not set"
            moving = "set" if pair.layer2_coords is not None else "not set"
            status = (
                f"Point picker: pair {pair.pair_id}, fixed {fixed}, "
                f"moving {moving}"
            )
            if not pair.enabled:
                status += " (skipped in fit)"
        report = self.fit_report
        if report is not None and report.ok:
            status += f" | fit RMS {report.rms:.2f} on {report.n_used} pairs"
        keys = "  ".join(
            f"{display} {description}"
            for _, display, _, description in KEYBINDINGS
        )
        return f"{status}\n{keys}"

    def cleanup(self) -> None:
        """Hide the canvas overlay and restore the moving layer's nudge."""
        self.set_canvas_overlay_active(False)
        self.cleanup_nudge()

    # ----------------------------------------------------- coordinate frames

    def _capture_translates(self) -> None:
        """Read and store the original translate vectors from both layers.

        Called lazily the first time coordinates are needed, since layers may
        not exist at __init__ time.
        """
        if self._translates_captured:
            return
        found = 0
        for name in (self.layer1_name, self.layer2_name):
            if name in self.viewer.layers:
                layer = self.viewer.layers[name]
                translate = np.array(layer.translate, dtype=float)
                self._original_translates[name] = translate
                origin = (layer.metadata or {}).get(FULL_RES_ORIGIN_KEY)
                self._centre_offsets[name] = (
                    translate - np.asarray(origin, dtype=float)
                    if origin is not None
                    else np.zeros_like(translate)
                )
                found += 1
            else:
                self._original_translates[name] = np.zeros(3)
                self._centre_offsets[name] = np.zeros(3)
        self._translates_captured = found == 2
        if self._translates_captured:
            self._update_layers_label()

    def _uses_full_res_units(self) -> bool:
        layers = self.viewer.layers
        return all(
            name in layers
            and FULL_RES_ORIGIN_KEY in (layers[name].metadata or {})
            for name in (self.layer1_name, self.layer2_name)
        )

    def _coordinate_space(self) -> str:
        if self._uses_full_res_units():
            return "full_resolution_voxels"
        return "world_minus_layer_translate"

    def _layer_origin(self, layer_name: str) -> np.ndarray:
        """World position of the layer's data-space origin."""
        self._capture_translates()
        translate = self._original_translates.get(layer_name, np.zeros(3))
        offset = self._centre_offsets.get(layer_name, np.zeros(3))
        return translate - offset

    def _transform_coords(self, coords: tuple, affine: np.ndarray) -> tuple:
        """Apply a homogeneous affine matrix to world coordinates."""
        ndim = affine.shape[0] - 1
        p = np.array(coords[:ndim], dtype=float)
        transformed = affine[:ndim, :ndim] @ p + affine[:ndim, ndim]
        result = list(coords)  # preserve any extra dims unchanged
        result[:ndim] = transformed.tolist()
        return tuple(result)

    def _get_base_translate(self) -> np.ndarray:
        """Return what layer2.translate would be without the nudge offset."""
        if self._affine_base_translate is not None:
            return self._affine_base_translate.copy()
        self._capture_translates()
        return self._original_translates.get(
            self.layer2_name, np.zeros(3)
        ).copy()

    def _world_to_original_data(
        self, coords: tuple | None, layer_name: str
    ) -> tuple | None:
        """Convert world coordinates to the layer's data space.

        Data space is world coordinates relative to the layer's origin: its
        original translate (captured before any padding or affine), or the
        full-resolution origin when the layer shows a downsampled level (see
        ``FULL_RES_ORIGIN_KEY``). With unit scale it indexes the original
        data array.
        """
        if coords is None:
            return None
        origin = self._layer_origin(layer_name)
        arr = np.array(coords[: len(origin)], dtype=float)
        arr -= origin
        return tuple(arr.tolist())

    def _original_data_to_world(self, coords: tuple, layer_name: str) -> tuple:
        """Convert data-space coordinates to world coordinates."""
        origin = self._layer_origin(layer_name)
        arr = np.array(coords[: len(origin)], dtype=float)
        arr += origin
        return tuple(arr.tolist())

    # ----------------------------------------------------------------- nudge

    def _update_layer2_translate(self) -> None:
        """Set layer2.translate to base + nudge_offset."""
        if self.layer2_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[self.layer2_name]
        base = self._get_base_translate()
        layer.translate = base + self._nudge_offset

    def _apply_nudge(self) -> None:
        """Called when any nudge spinbox value changes."""
        self._capture_translates()
        self._nudge_offset = np.array(
            [spin.value() for spin in self._nudge_spins]
        )
        self._update_layer2_translate()

    def _reset_nudge(self) -> None:
        """Reset all nudge spinboxes to zero."""
        for spin in self._nudge_spins:
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._nudge_offset = np.zeros(3)
        self._reset_nudge_button.setStyleSheet("")
        self._update_layer2_translate()

    def cleanup_nudge(self) -> None:
        """Reset the nudge offset and restore layer2.translate to its base value."""
        if not np.allclose(self._nudge_offset, 0):
            self._reset_nudge()

    # ------------------------------------------------------ editing pairs

    def _push_undo(self) -> None:
        state = (
            [replace(p) for p in self.point_pairs.values()],
            self._next_pair_id,
            self._active_pair_id,
        )
        self._undo_stack.append(state)
        del self._undo_stack[:-UNDO_LIMIT]

    def undo(self) -> None:
        """Undo the last change to the pairs (add, set, delete, use, load)."""
        if not self._undo_stack:
            show_info("Point picker: nothing to undo.")
            return
        pairs, next_id, active = self._undo_stack.pop()
        self.point_pairs = {p.pair_id: p for p in pairs}
        self._next_pair_id = next_id
        self._active_pair_id = active
        self._pairs_changed(structure=True)

    def add_pair(self, *, record_undo: bool = True) -> int:
        """Add an empty pair, make it the active pair, and return its id."""
        if record_undo:
            self._push_undo()
        pair_id = self._next_pair_id
        self._next_pair_id += 1
        self.point_pairs[pair_id] = PointPair(pair_id=pair_id)
        self._active_pair_id = pair_id
        self._pairs_changed(structure=True)
        return pair_id

    @property
    def active_pair_id(self) -> int | None:
        return self._active_pair_id

    def set_active_pair(self, pair_id: int | None) -> None:
        if pair_id is not None and pair_id not in self.point_pairs:
            return
        self._active_pair_id = pair_id
        self._select_active_row()
        self._update_canvas_overlay()

    def record_fixed(self) -> None:
        """Store the crosshair position as the active pair's fixed point."""
        self._record_active("layer1")

    def record_moving(self) -> None:
        """Store the crosshair position as the active pair's moving point."""
        self._record_active("layer2")

    def _record_active(self, layer: str) -> None:
        self._push_undo()
        if self._active_pair_id not in self.point_pairs:
            self.add_pair(record_undo=False)
        self._update_coordinate(
            self._active_pair_id, layer, record_undo=False
        )

    def go_to_fixed(self) -> None:
        if self._active_pair_id is not None:
            self._show_coordinate(self._active_pair_id, "layer1")

    def go_to_moving(self) -> None:
        if self._active_pair_id is not None:
            self._show_coordinate(self._active_pair_id, "layer2")

    def select_next_pair(self) -> None:
        self._step_active(+1)

    def select_previous_pair(self) -> None:
        self._step_active(-1)

    def _step_active(self, step: int) -> None:
        order = self._display_order()
        if not order:
            return
        if self._active_pair_id in order:
            index = order.index(self._active_pair_id) + step
            index = min(max(index, 0), len(order) - 1)
        else:
            index = 0 if step > 0 else len(order) - 1
        self.set_active_pair(order[index])

    def toggle_active_use(self) -> None:
        """Include or exclude the active pair from the fit."""
        pair = self.point_pairs.get(self._active_pair_id)
        if pair is not None:
            self.set_pair_enabled(pair.pair_id, not pair.enabled)

    def set_pair_enabled(self, pair_id: int, enabled: bool) -> None:
        pair = self.point_pairs.get(pair_id)
        if pair is None or pair.enabled == enabled:
            return
        self._push_undo()
        pair.enabled = enabled
        self._pairs_changed()

    def _show_coordinate(self, pair_id: int, layer: str) -> None:
        """Move crosshair to the saved world coordinate for the given pair and layer."""

        if pair_id not in self.point_pairs:
            return

        self._capture_translates()

        point_pair = self.point_pairs[pair_id]
        coords = (
            point_pair.layer1_coords
            if layer == "layer1"
            else point_pair.layer2_coords
        )

        if coords is None:
            side = "fixed" if layer == "layer1" else "moving"
            show_info(f"Point picker: pair {pair_id} has no {side} point yet.")
            return

        if layer == "layer2":
            if self._applied_affine is not None:
                coords = self._transform_coords(coords, self._applied_affine)
            if not np.allclose(self._nudge_offset, 0):
                coords = tuple(
                    (np.array(coords) + self._nudge_offset).tolist()
                )

        self._active_pair_id = pair_id
        self._select_active_row()
        self.viewer.dims.point = coords

    def _update_coordinate(
        self, pair_id: int, layer: str, *, record_undo: bool = True
    ) -> None:
        """Save the current crosshair world position for the given pair and layer."""

        if pair_id not in self.point_pairs:
            return

        self._capture_translates()

        current_coords = tuple(self.viewer.dims.point)

        if layer == "layer2":
            if not np.allclose(self._nudge_offset, 0):
                current_coords = tuple(
                    (np.array(current_coords) - self._nudge_offset).tolist()
                )
            if self._applied_affine is not None:
                inv = np.linalg.inv(self._applied_affine)
                current_coords = self._transform_coords(current_coords, inv)

        if record_undo:
            self._push_undo()
        point_pair = self.point_pairs[pair_id]
        if layer == "layer1":
            point_pair.layer1_coords = current_coords
        else:
            point_pair.layer2_coords = current_coords
        self._active_pair_id = pair_id
        self._pairs_changed()

    def _delete_pair(self, pair_id: int) -> None:
        """Delete the point pair with the given pair_id (undoable)."""

        if pair_id not in self.point_pairs:
            return
        self._push_undo()
        if self._active_pair_id == pair_id:
            order = self._display_order()
            index = order.index(pair_id)
            remaining = order[:index] + order[index + 1 :]
            self._active_pair_id = (
                remaining[min(index, len(remaining) - 1)]
                if remaining
                else None
            )
        del self.point_pairs[pair_id]
        self._pairs_changed(structure=True)

    def clear_pairs(self) -> None:
        """Clear all point pairs."""

        self.point_pairs.clear()
        self._next_pair_id = 1
        self._active_pair_id = None
        self._pairs_changed(structure=True)

    def _pairs_changed(self, structure: bool = False) -> None:
        """Refresh everything that depends on the pairs after an edit."""
        if structure:
            self._rebuild_table()
        else:
            self._refresh_rows()
        self._fit_timer.start()
        self._autosave()
        self._update_button_states()
        self._update_canvas_overlay()

    # ----------------------------------------------------------- the table

    def _complete_pairs(self) -> dict[int, PointPair]:
        return {
            pid: p
            for pid, p in self.point_pairs.items()
            if p.layer1_coords is not None and p.layer2_coords is not None
        }

    def _display_order(self) -> list[int]:
        ids = list(self.point_pairs)
        mode = self.order_combo.currentIndex()
        report = self.fit_report
        if mode == ORDER_PAIR or report is None:
            return ids
        values = (
            report.residual_norms
            if mode == ORDER_RESIDUAL
            else report.out_of_sample
        )
        return sorted(ids, key=lambda i: -values.get(i, -1.0))

    def _rebuild_table(self) -> None:
        self._building_table = True
        try:
            self.table.setRowCount(0)
            self._row_cells.clear()
            for row, pair_id in enumerate(self._display_order()):
                self._insert_row(row, pair_id)
            self.table.resizeRowsToContents()
        finally:
            self._building_table = False
        self._refresh_rows()
        self._select_active_row()

    def _insert_row(self, row: int, pair_id: int) -> None:
        self.table.insertRow(row)
        pair_item = QTableWidgetItem(str(pair_id))
        pair_item.setFlags(pair_item.flags() & ~Qt.ItemIsEditable)
        pair_item.setData(Qt.UserRole, pair_id)
        pair_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, COL_PAIR, pair_item)

        use_item = QTableWidgetItem()
        use_item.setFlags(
            Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        )
        use_item.setData(Qt.UserRole, pair_id)
        self.table.setItem(row, COL_USE, use_item)

        cells = []
        for column, layer in ((COL_FIXED, "layer1"), (COL_MOVING, "layer2")):
            cell = ShowUpdateWidget()
            cell.show_clicked.connect(
                lambda pid=pair_id, lyr=layer: self._show_coordinate(pid, lyr)
            )
            cell.update_clicked.connect(
                lambda pid=pair_id, lyr=layer: self._update_coordinate(
                    pid, lyr
                )
            )
            self.table.setCellWidget(row, column, cell)
            cells.append(cell)
        self._row_cells[pair_id] = tuple(cells)

        for column in (COL_RESID, COL_LOO):
            item = QTableWidgetItem("")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, column, item)

        delete_button = QPushButton("×")
        delete_button.setMaximumWidth(30)
        delete_button.setToolTip("Delete this pair (Ctrl+Z undoes it).")
        delete_button.clicked.connect(
            lambda checked=False, pid=pair_id: self._delete_pair(pid)
        )
        self.table.setCellWidget(row, COL_DELETE, delete_button)

    def _row_of(self, pair_id: int) -> int | None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_PAIR)
            if item is not None and item.data(Qt.UserRole) == pair_id:
                return row
        return None

    def _refresh_rows(self) -> None:
        """Update coordinates, residuals and highlighting in place."""
        report = self.fit_report
        threshold = self.flag_spin.value()
        self._building_table = True
        try:
            for row in range(self.table.rowCount()):
                pair_id = self.table.item(row, COL_PAIR).data(Qt.UserRole)
                pair = self.point_pairs.get(pair_id)
                if pair is None:
                    continue
                fixed_cell, moving_cell = self._row_cells[pair_id]
                fixed_cell.set_coords(
                    self._world_to_original_data(
                        pair.layer1_coords, self.layer1_name
                    )
                )
                moving_cell.set_coords(
                    self._world_to_original_data(
                        pair.layer2_coords, self.layer2_name
                    )
                )
                self.table.item(row, COL_USE).setCheckState(
                    Qt.Checked if pair.enabled else Qt.Unchecked
                )
                resid = loo = None
                if report is not None:
                    resid = report.residual_norms.get(pair_id)
                    loo = report.out_of_sample.get(pair_id)
                flagged = any(
                    v is not None and v > threshold for v in (resid, loo)
                )
                for column, value in ((COL_RESID, resid), (COL_LOO, loo)):
                    item = self.table.item(row, column)
                    item.setText("" if value is None else f"{value:.2f}")
                    item.setForeground(
                        QBrush(EXCLUDED_COLOUR)
                        if not pair.enabled
                        else QBrush()
                    )
                for column in (COL_PAIR, COL_RESID, COL_LOO):
                    self.table.item(row, column).setBackground(
                        QBrush(FLAG_COLOUR) if flagged else QBrush()
                    )
                tip = "" if pair.enabled else "Not used in the fit."
                self.table.item(row, COL_PAIR).setToolTip(tip)
        finally:
            self._building_table = False

    def _select_active_row(self) -> None:
        row = (
            self._row_of(self._active_pair_id)
            if self._active_pair_id is not None
            else None
        )
        self._building_table = True
        try:
            if row is None:
                self.table.clearSelection()
            else:
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, COL_PAIR))
        finally:
            self._building_table = False

    def _on_selection_changed(self) -> None:
        if self._building_table:
            return
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return
        item = self.table.item(rows.pop(), COL_PAIR)
        if item is not None:
            self._active_pair_id = item.data(Qt.UserRole)
            self._update_canvas_overlay()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building_table or item.column() != COL_USE:
            return
        self.set_pair_enabled(
            item.data(Qt.UserRole), item.checkState() == Qt.Checked
        )

    # ------------------------------------------------------------- the fit

    def _moving_world_corners(self) -> np.ndarray | None:
        """World positions of the moving image's corner voxels (pre-affine)."""
        if self.layer2_name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[self.layer2_name]
        if self.transform_snapshot is not None:
            shape = self.transform_snapshot["data"].shape
            translate = self.transform_snapshot["translate"]
        else:
            shape = layer.data.shape
            translate = self._get_base_translate()
        ndim = len(shape)
        scale = np.asarray(layer.scale, dtype=float)[:ndim]
        translate = np.asarray(translate, dtype=float)[:ndim]
        if len(translate) != ndim:
            return None
        corners = np.array(
            list(itertools.product(*[(0, s - 1) for s in shape])),
            dtype=float,
        )
        return corners * scale + translate

    def _pick_sigma(self) -> float:
        """Simulated pick noise: half a displayed voxel of the fixed layer."""
        if self.layer1_name in self.viewer.layers:
            scale = self.viewer.layers[self.layer1_name].scale
            return 0.5 * float(np.max(scale))
        return 0.5

    def refresh_fit(self) -> FitReport:
        """Re-estimate the fit from the current pairs and update the panel."""
        self._fit_timer.stop()
        complete = self._complete_pairs()
        report = evaluate_fit(
            self.affine_estimator,
            {pid: p.layer1_coords for pid, p in complete.items()},
            {pid: p.layer2_coords for pid, p in complete.items()},
            [pid for pid, p in complete.items() if p.enabled],
            volume_corners=self._moving_world_corners(),
            pick_sigma=self._pick_sigma(),
        )
        self.fit_report = report
        if self.order_combo.currentIndex() != ORDER_PAIR:
            self._rebuild_table()
        else:
            self._refresh_rows()
        self.fit_label.setText(self._format_fit_report(report))
        self._update_button_states()
        self._update_canvas_overlay()
        self.fit_updated.emit(report)
        return report

    def _format_fit_report(self, r: FitReport) -> str:
        units = self._units_phrase()
        excluded = r.n_complete - r.n_used
        excluded_note = (
            f" ({excluded} complete pair{'s' if excluded != 1 else ''} "
            "skipped)"
            if excluded
            else ""
        )
        if r.n_used < r.min_pairs:
            need = r.pairs_needed
            return (
                f"<b>Need {need} more complete pair"
                f"{'s' if need != 1 else ''} in the fit</b>: "
                f"{r.n_used} of the {r.min_pairs} the estimator needs"
                f"{excluded_note}."
            )
        if r.error is not None:
            return (
                "<b style='color:#e74c3c'>The fit failed:</b> "
                f"{html.escape(r.error)}"
            )
        lines = [
            (
                f"<b>Fit on {r.n_used} pairs</b>{excluded_note}: RMS "
                f"{r.rms:.2f}, max {r.residual_norms[r.max_pair]:.2f} "
                f"(pair {r.max_pair}), in {units}."
            )
        ]
        if r.loo_rms is not None:
            worst = r.out_of_sample[r.worst_loo_pair]
            lines.append(
                f"Leave-one-out: RMS {r.loo_rms:.2f}, worst pair "
                f"{r.worst_loo_pair} ({worst:.2f})."
            )
        else:
            lines.append(
                f"Leave-one-out residuals need {r.min_pairs + 1} pairs in "
                "the fit."
            )
        if r.stability is not None:
            lines.append(
                "Stability: with simulated pick noise of "
                f"{r.stability_sigma:.2g} per axis, the corners of the moving "
                f"image move by up to {r.stability:.2f}."
            )
        if r.coverage is not None and len(r.coverage) == 3:
            spread = ", ".join(
                f"{name} {100 * c:.0f}%"
                for name, c in zip(AXIS_NAMES_3D, r.coverage, strict=True)
            )
            lines.append(f"Picks span {spread} of the moving image.")
        if r.summary is not None:
            s = r.summary
            lines.append(
                "Scale (z, y, x) "
                + ", ".join(f"{v:.4f}" for v in s.scale)
                + f"; rotation {s.rotation_deg:.3f} deg; shear "
                + ", ".join(f"{v:.4f}" for v in s.shear)
                + "; translation "
                + ", ".join(f"{v:.2f}" for v in s.translation)
                + "."
            )
        for warning in r.warnings:
            lines.append(
                f"<span style='color:#e67e22'><b>Warning:</b> "
                f"{html.escape(warning)}</span>"
            )
        return "<br>".join(lines)

    def _update_button_states(self) -> None:
        """Enable buttons that have something to act on, and say why not."""
        report = self.fit_report
        can_fit = report is not None and report.ok
        if report is None:
            reason = ""
        elif report.n_used < report.min_pairs:
            reason = (
                f"Needs {report.pairs_needed} more complete pair(s) in the "
                "fit."
            )
        elif report.error is not None:
            reason = f"The fit failed: {report.error}"
        else:
            reason = ""
        self.apply_button.setEnabled(can_fit)
        self.apply_button.setToolTip(reason)
        self.preview_estimate_button.setEnabled(can_fit)
        self.preview_estimate_button.setToolTip(
            reason
            or "The moving image mapped through the current fit (applied "
            "or not), with the picks."
        )

        has_snapshot = self.transform_snapshot is not None
        self.reset_button.setEnabled(has_snapshot)
        self.restart_button.setEnabled(has_snapshot)
        self.undo_button.setEnabled(bool(self._undo_stack))

    def _estimate_affine_transform(self) -> np.ndarray | None:
        """Estimate the world-space affine from the pairs used in the fit.

        Returns:
            Homogeneous affine matrix (4x4 for 3D, 3x3 for 2D) or None if insufficient pairs.
        """
        used = [p for p in self._complete_pairs().values() if p.enabled]
        if not used:
            return None
        ndim = len(used[0].layer2_coords)
        if len(used) < ndim + 1:
            return None

        # Transform the moving layer to match the fixed layer:
        # - Source points are from the moving layer (layer2_coords)
        # - Target points are from the fixed layer (layer1_coords)
        src_pts = np.array([p.layer2_coords for p in used])
        tgt_pts = np.array([p.layer1_coords for p in used])
        return self.affine_estimator(src_pts, tgt_pts)

    def _print_data_space_residuals(self, data_affine: np.ndarray) -> None:
        """Print per-pair residuals for the estimated data-space affine."""
        used = [p for p in self._complete_pairs().values() if p.enabled]
        if not used:
            return
        ndim = data_affine.shape[0] - 1
        src = np.array(
            [
                self._world_to_original_data(p.layer2_coords, self.layer2_name)
                for p in used
            ]
        )[:, :ndim]
        tgt = np.array(
            [
                self._world_to_original_data(p.layer1_coords, self.layer1_name)
                for p in used
            ]
        )[:, :ndim]
        residuals = apply_affine(data_affine, src) - tgt
        norms = np.linalg.norm(residuals, axis=1)
        rms = float(np.sqrt(np.mean(norms**2)))
        worst = used[int(np.argmax(norms))].pair_id
        print(
            "[PointPicker] Data-space residuals after affine "
            f"(RMS={rms:.3f}, max={norms.max():.3f} at pair {worst}):"
        )
        for pair, residual, norm in zip(used, residuals, norms, strict=True):
            values = ", ".join(f"{value:.3f}" for value in residual)
            print(
                f"  pair {pair.pair_id}: residual=({values}), "
                f"norm={float(norm):.3f}"
            )

    def get_estimated_affine(self) -> np.ndarray | None:
        """Get the estimated affine in data coordinates.

        The internal affine maps layer2-world → layer1-world.  This method
        conjugates it so that it operates in each layer's data space (see
        ``_world_to_original_data``):

            A_data = T1_inv @ A_world @ T2

        where T2 translates by +layer2_origin and T1_inv by -layer1_origin.

        Returns:
            Homogeneous affine matrix or None if insufficient valid pairs.
        """
        affine = self._estimate_affine_transform()
        if affine is None:
            return affine

        ndim = affine.shape[0] - 1
        t1 = self._layer_origin(self.layer1_name)[:ndim]
        t2 = self._layer_origin(self.layer2_name)[:ndim]

        # If both origins are zero (no offset), skip conjugation
        if np.allclose(t1, 0) and np.allclose(t2, 0):
            return affine

        T2 = np.eye(ndim + 1)
        T2[:ndim, ndim] = t2
        T1_inv = np.eye(ndim + 1)
        T1_inv[:ndim, ndim] = -t1
        return T1_inv @ affine @ T2

    # ----------------------------------------------------- apply / reset

    def _apply_affine(self) -> None:
        """Apply the estimated affine transform to the moving layer.

        Pre-transforms the image data rather than setting layer.affine,
        because napari does not fully support non-orthogonal slicing:
        off-diagonal affine components (shear/rotation) are stripped in
        non-displayed dimensions, causing orthoviews to render incorrectly.

        The output array keeps the layer's scale and is sized to fit the
        axis-aligned bounding box of layer2's transformed corners, so no
        image data is clipped. The layer's translate is set to the bounding
        box origin so that the result sits at the correct world position.
        """
        if self.layer2_name not in self.viewer.layers:
            return
        try:
            self._apply_affine_unchecked()
        except Exception as exc:  # noqa: BLE001 (shown to the user)
            traceback.print_exc()
            show_error(
                f"Could not apply the affine: {type(exc).__name__}: {exc}"
            )

    def _apply_affine_unchecked(self) -> None:
        self._capture_translates()
        layer = self.viewer.layers[self.layer2_name]

        affine = self._estimate_affine_transform()
        if affine is None:
            show_warning(
                "Point picker: not enough complete pairs in the fit to "
                "apply an affine."
            )
            return
        ndim = affine.shape[0] - 1
        A_lin = affine[:ndim, :ndim]
        A_trans = affine[:ndim, ndim]
        A_lin_inv = np.linalg.inv(A_lin)

        # Snapshot current state if not already done
        if self.transform_snapshot is None:
            self.transform_snapshot = {
                "affine": layer.affine.affine_matrix.copy(),
                "data": layer.data,
                "translate": np.array(layer.translate, dtype=float)
                - self._nudge_offset,
            }

        pairs = self.get_point_pairs()
        np.set_printoptions(suppress=True, precision=10, linewidth=200)
        print("[PointPicker] Selected point pairs (data space):")
        used_ids = [
            p.pair_id for p in self._complete_pairs().values() if p.enabled
        ]
        for pair_id, l1_pt, l2_pt in zip(
            used_ids,
            pairs[self.layer1_name],
            pairs[self.layer2_name],
            strict=True,
        ):
            print(
                f"  pair {pair_id}: {self.layer1_name}: {l1_pt}   "
                f"{self.layer2_name}: {l2_pt}"
            )
        print("[PointPicker] World-space affine (layer2 → layer1):")
        print(affine)
        data_affine = self.get_estimated_affine()
        print("[PointPicker] Data-space affine (layer2 → layer1):")
        print(data_affine)
        self._print_data_space_residuals(data_affine)

        # Layer2's original translate (not current; current may be bbox_min
        # from a previous application)
        T2 = self.transform_snapshot["translate"][:ndim]
        scale = np.asarray(layer.scale, dtype=float)[:ndim]
        data_shape = np.array(
            self.transform_snapshot["data"].shape[:ndim], dtype=float
        )

        # Build the 2^ndim corners of layer2 in world space
        corners_data = np.array(
            list(itertools.product(*[(0, s) for s in data_shape]))
        )
        corners_world = corners_data * scale + T2

        # Transform corners through the affine (layer2-world → layer1-world)
        corners_transformed = corners_world @ A_lin.T + A_trans

        # Axis-aligned bounding box of the transformed volume, snapped to
        # the layer's own voxel grid so the output stays aligned with it.
        lo = np.floor((corners_transformed.min(axis=0) - T2) / scale)
        hi = np.ceil((corners_transformed.max(axis=0) - T2) / scale)
        bbox_min = T2 + lo * scale
        output_shape = tuple((hi - lo).astype(int).tolist())

        # Output index o sits at world w = scale * o + bbox_min. Its source
        # in layer2-world is A_lin_inv @ (w - A_trans), and its layer2 data
        # index is (that - T2) / scale, so
        #   d2 = S^-1 A_lin_inv S o + S^-1 (A_lin_inv (bbox_min - A_trans) - T2)
        inv_scale = np.diag(1.0 / scale)
        affiners_matrix = np.eye(ndim + 1)
        affiners_matrix[:ndim, :ndim] = inv_scale @ A_lin_inv @ np.diag(scale)
        affiners_matrix[:ndim, ndim] = inv_scale @ (
            A_lin_inv @ (bbox_min - A_trans) - T2
        )
        import torch

        data = (
            torch.from_numpy(np.asarray(self.transform_snapshot["data"]))
            .to(torch.float16)
            .numpy()
        )
        transformed = affiners.affine_transform(
            data,
            affiners_matrix,
            output_shape=output_shape,
        )
        layer.data = transformed

        # Restore original affine (identity in most cases) so no
        # non-orthogonal slicing occurs.
        layer.affine = self.transform_snapshot["affine"]

        # Place the output at the bounding box origin in world space,
        # preserving any active nudge offset.
        self._affine_base_translate = bbox_min.astype(float)
        layer.translate = self._affine_base_translate + self._nudge_offset

        self._applied_affine = affine
        self.affine_applied.emit(affine)
        self._update_button_states()

        if not np.allclose(self._nudge_offset, 0):
            self._reset_nudge_button.setStyleSheet(
                "QPushButton { background-color: #c0392b; color: white; }"
            )
            show_warning(
                "Affine applied while nudge is active. "
                "Consider resetting the nudge offset to see the "
                "aligned result without the extra translation."
            )

    def _reset_transform(self) -> None:
        """Reset the transform of the moving layer to the snapshot state."""

        if self.transform_snapshot is None:
            return

        if self.layer2_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[self.layer2_name]
        layer.data = self.transform_snapshot["data"]
        layer.affine = self.transform_snapshot["affine"]
        layer.translate = (
            self.transform_snapshot["translate"] + self._nudge_offset
        )
        self.transform_snapshot = None
        self._applied_affine = None
        self._affine_base_translate = None
        self._update_button_states()

    def _restart_from_current(self) -> None:
        """Drop point pairs and transform state but keep the moving layer's
        currently displayed (transformed) image in place.

        Re-baselines layer2 so its current world position (minus the active
        nudge offset) becomes the new "original" translate. Subsequent picks
        and Apply operations then treat the displayed view as the starting
        point, enabling iterative refinement: apply -> restart -> pick new
        correspondences on the partially-aligned image -> apply again.
        This cannot be undone; the pairs are still in the autosave file.
        """
        if self.transform_snapshot is None:
            return
        if self.layer2_name not in self.viewer.layers:
            return

        self._capture_translates()
        layer = self.viewer.layers[self.layer2_name]
        # The displayed translate includes the nudge; strip it so the stored
        # baseline matches what _apply_affine snapshots ("base", not "base + nudge").
        self._original_translates[self.layer2_name] = (
            np.array(layer.translate, dtype=float) - self._nudge_offset
        )

        self._undo_stack.clear()
        # Start a new autosave file so the old pairs stay recoverable.
        self.autosave_path = None
        self.backup_path = None
        self.transform_snapshot = None
        self._applied_affine = None
        self._affine_base_translate = None
        self._rebased = True
        self.clear_pairs()

    # ------------------------------------------------- programmatic access

    def get_point_pairs(self, include_skipped: bool = False) -> dict:
        """Return complete point pairs as two parallel coordinate lists.

        Coordinates are in each layer's data space (see
        ``_world_to_original_data``), matching the format expected by
        ``load_point_pairs()``.

        Parameters
        ----------
        include_skipped : bool
            Also return pairs whose "Use" box is unchecked.

        Returns:
            Dict keyed by layer name, each a list of (z, y, x) tuples.
        """
        layer1 = []
        layer2 = []
        for pair in self._complete_pairs().values():
            if not (pair.enabled or include_skipped):
                continue
            layer1.append(
                self._world_to_original_data(
                    pair.layer1_coords, self.layer1_name
                )
            )
            layer2.append(
                self._world_to_original_data(
                    pair.layer2_coords, self.layer2_name
                )
            )
        return {self.layer1_name: layer1, self.layer2_name: layer2}

    def load_point_pairs(self, pairs: dict) -> None:
        """Load point pairs into the widget (undoable).

        Parameters
        ----------
        pairs : dict
            Dict keyed by layer name, each a list of (z, y, x) coordinate
            tuples in each layer's data space.  Matches the format
            returned by ``get_point_pairs()``.
        """
        records = [
            {"fixed": l1, "moving": l2, "use_in_fit": True}
            for l1, l2 in zip(
                pairs[self.layer1_name], pairs[self.layer2_name], strict=True
            )
        ]
        self._load_records(records)

    def _load_records(self, records: list[dict]) -> None:
        self._capture_translates()
        self._push_undo()
        self.point_pairs = {}
        self._next_pair_id = 1
        for record in records:
            pair_id = int(record.get("pair", self._next_pair_id))
            self._next_pair_id = max(self._next_pair_id, pair_id + 1)
            fixed = record.get("fixed")
            moving = record.get("moving")
            self.point_pairs[pair_id] = PointPair(
                pair_id=pair_id,
                layer1_coords=(
                    None
                    if fixed is None
                    else self._original_data_to_world(
                        tuple(fixed), self.layer1_name
                    )
                ),
                layer2_coords=(
                    None
                    if moving is None
                    else self._original_data_to_world(
                        tuple(moving), self.layer2_name
                    )
                ),
                enabled=bool(record.get("use_in_fit", True)),
            )
        self._active_pair_id = next(iter(self.point_pairs), None)
        self._pairs_changed(structure=True)

    # --------------------------------------------------------- save / load

    def _layer_source_info(self, name: str) -> dict:
        if name not in self.viewer.layers:
            return {}
        layer = self.viewer.layers[name]
        data = layer.data
        if name == self.layer2_name and self.transform_snapshot is not None:
            data = self.transform_snapshot["data"]
        scale = np.asarray(layer.scale, dtype=float)
        info = {
            "data_shape": [int(s) for s in data.shape],
            "scale": scale.tolist(),
            "metadata": _json_safe(dict(layer.metadata or {})),
        }
        if FULL_RES_ORIGIN_KEY in (layer.metadata or {}):
            info["full_resolution_shape"] = [
                int(round(s * f))
                for s, f in zip(data.shape, scale, strict=False)
            ]
        if name == self.layer2_name and self._rebased:
            info["rebased_on_partial_alignment"] = True
        return info

    def build_save_payload(self) -> dict:
        """Everything needed to restore and audit the current picks."""
        pairs = self.get_point_pairs()
        records = []
        for pair in self.point_pairs.values():
            fixed = self._world_to_original_data(
                pair.layer1_coords, self.layer1_name
            )
            moving = self._world_to_original_data(
                pair.layer2_coords, self.layer2_name
            )
            records.append(
                {
                    "pair": pair.pair_id,
                    "fixed": None if fixed is None else list(fixed),
                    "moving": None if moving is None else list(moving),
                    "use_in_fit": pair.enabled,
                }
            )
        return {
            "format_version": PICK_FILE_VERSION,
            "saved_at": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "coordinate_space": self._coordinate_space(),
            "estimator": _describe_estimator(self.affine_estimator),
            "layer1_name": self.layer1_name,
            "layer2_name": self.layer2_name,
            "sources": {
                self.layer1_name: self._layer_source_info(self.layer1_name),
                self.layer2_name: self._layer_source_info(self.layer2_name),
            },
            # Complete pairs used in the fit, in the format older readers
            # (and get_point_pairs) use.
            "pairs": {
                self.layer1_name: [list(c) for c in pairs[self.layer1_name]],
                self.layer2_name: [list(c) for c in pairs[self.layer2_name]],
            },
            # Every pair, including unfinished and skipped ones.
            "pair_records": records,
        }

    def save_points(self, path: str | Path) -> Path:
        path = Path(path)
        _write_json_atomic(path, self.build_save_payload())
        show_info(
            f"Point picker: saved {len(self.point_pairs)} pairs to {path}"
        )
        return path

    def _save_points_dialog(self) -> None:
        """Prompt for a path and write the picks as JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save point pairs",
            "point_pairs.json",
            "JSON (*.json);;All files (*)",
        )
        if path:
            self.save_points(path)

    def _load_points_dialog(self) -> None:
        """Prompt for a JSON file and load point pairs from it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load point pairs",
            str(self.autosave_dir) if self.autosave_dir is not None else "",
            "JSON (*.json);;All files (*)",
        )
        if path:
            self.load_points(path)

    def load_points(self, path: str | Path) -> list[str]:
        """Load picks from a file written by Save Points or autosave.

        Also reads the older format (layer names plus complete pairs only).
        Returns the warnings shown about the file not matching the data
        that is open now.
        """
        path = Path(path)
        with path.open() as fh:
            payload = json.load(fh)

        if "pairs" in payload and isinstance(payload["pairs"], dict):
            src_l1 = payload.get("layer1_name")
            src_l2 = payload.get("layer2_name")
        else:
            payload = {"pairs": payload}
            src_l1, src_l2 = self.layer1_name, self.layer2_name

        if "pair_records" in payload:
            records = payload["pair_records"]
        else:
            pairs_raw = payload["pairs"]
            key1 = src_l1 if src_l1 in pairs_raw else self.layer1_name
            key2 = src_l2 if src_l2 in pairs_raw else self.layer2_name
            records = [
                {"pair": i, "fixed": l1, "moving": l2, "use_in_fit": True}
                for i, (l1, l2) in enumerate(
                    zip(pairs_raw[key1], pairs_raw[key2], strict=True),
                    start=1,
                )
            ]

        warnings = self._compare_sources(payload, src_l1, src_l2)
        self._load_records(records)
        if warnings:
            show_warning(
                f"Point picker: {path.name} may not match the open data. "
                + " ".join(warnings)
            )
        show_info(f"Point picker: loaded {len(records)} pairs from {path}")
        return warnings

    def _compare_sources(self, payload: dict, src_l1, src_l2) -> list[str]:
        warnings = []
        here = self._coordinate_space()
        there = payload.get("coordinate_space")
        if there is None:
            if here == "full_resolution_voxels":
                warnings.append(
                    "The file predates recorded coordinate frames, so its "
                    "coordinates are in the voxels of whichever pyramid "
                    "level was open when it was saved, while the open "
                    "layers use full-resolution voxels. Rescale it first if "
                    "it was picked on a downsampled level."
                )
            return warnings
        if there != here:
            warnings.append(
                f"The file's coordinates are in {there.replace('_', ' ')} "
                f"but the open layers use {here.replace('_', ' ')}."
            )
        sources = payload.get("sources", {})
        for role, file_name, name in (
            ("fixed", src_l1, self.layer1_name),
            ("moving", src_l2, self.layer2_name),
        ):
            saved = sources.get(file_name)
            if not saved:
                continue
            current = self._layer_source_info(name)
            saved_meta = saved.get("metadata") or {}
            current_meta = current.get("metadata") or {}
            keys = ["timepoint", "channel", "y_flipped", "x_flipped"]
            keys.insert(
                0,
                "source_zarr_root"
                if "source_zarr_root" in saved_meta
                or "source_zarr_root" in current_meta
                else "source_zarr",
            )
            for key in keys:
                if (
                    key in saved_meta
                    and key in current_meta
                    and saved_meta[key] != current_meta[key]
                ):
                    warnings.append(
                        f"The {role} layer's {key} was "
                        f"{saved_meta[key]!r} in the file and is "
                        f"{current_meta[key]!r} now."
                    )
            a = saved.get("full_resolution_shape")
            b = current.get("full_resolution_shape")
            if a and b and a != b:
                warnings.append(
                    f"The {role} layer's full-resolution shape was {a} in "
                    f"the file and is {b} now."
                )
        return warnings

    def _picks_file_name(self) -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.layer2_name)
        return f"{stamp}_{slug.strip('-')[:40] or 'picks'}.json"

    def _autosave(self) -> None:
        """Write the picks after an edit: to the autosave file, and a copy
        to the backup folder in /tmp, which is written even when the
        autosave location or the Save dialog fails."""
        if (
            self.autosave_path is None
            and self.point_pairs
            and self.autosave_dir is not None
        ):
            self.autosave_path = self.autosave_dir / self._picks_file_name()
        if (
            self.backup_path is None
            and self.point_pairs
            and self.backup_dir is not None
        ):
            self.backup_path = self.backup_dir / self._picks_file_name()
        targets = [
            p for p in (self.autosave_path, self.backup_path) if p is not None
        ]
        if not targets:
            return
        payload = self.build_save_payload()
        written = []
        for path in targets:
            try:
                _write_json_atomic(path, payload)
            except OSError as exc:
                if path not in self._save_failures:
                    show_warning(
                        f"Point picker: could not save picks to {path}: {exc}"
                    )
                    self._save_failures.add(path)
                continue
            written.append(path)
        if written:
            self.autosave_label.setText(
                "Saved after every edit to "
                + " and ".join(str(p) for p in written)
            )

    # ---------------------------------------------------------- 3D preview

    def _preview_jobs(self) -> list[dict]:
        jobs = []
        for role, name in (
            ("fixed", self.layer1_name),
            ("moving", self.layer2_name),
        ):
            if name not in self.viewer.layers:
                raise ValueError(f"Layer {name!r} is not in the viewer.")
            layer = self.viewer.layers[name]
            if role == "moving" and self.transform_snapshot is not None:
                data = self.transform_snapshot["data"]
                translate = self.transform_snapshot["translate"]
            elif role == "moving":
                data = layer.data
                translate = self._get_base_translate()
            else:
                data = layer.data
                translate = np.array(layer.translate, dtype=float)
            if data.ndim != 3:
                raise ValueError(
                    "The 3D preview needs 3D layers; "
                    f"{name!r} has {data.ndim} dimensions."
                )
            jobs.append(
                {
                    "role": role,
                    "name": name,
                    "data": data,
                    "scale": np.asarray(layer.scale, dtype=float),
                    "translate": np.asarray(translate, dtype=float),
                    "colormap": layer.colormap,
                    "contrast_limits": tuple(layer.contrast_limits),
                    "gamma": layer.gamma,
                    # After "Reset (keep view)" the displayed moving image no
                    # longer matches its source, so stride it instead.
                    "use_provider": not (role == "moving" and self._rebased),
                }
            )
        return jobs

    def _load_preview_volume(self, job: dict, max_size: int):
        provider = self.preview_data_provider
        if job["use_provider"] and provider is not None:
            result = provider(job["name"], max_size)
            if result is not None:
                array, scale, translate = result
                return (
                    _finite(np.asarray(array)),
                    np.asarray(scale, dtype=float),
                    np.asarray(translate, dtype=float),
                )
        return downsample_for_preview(
            job["data"], job["scale"], job["translate"], max_size
        )

    def open_3d_preview(self, with_estimate: bool = False) -> None:
        """Open a 3D viewer on downsampled copies of both layers.

        With ``with_estimate`` the moving image is mapped through the
        current fit (whether or not it has been applied). Loading runs in a
        background thread; the window opens when it finishes.
        """
        try:
            jobs = self._preview_jobs()
        except ValueError as exc:
            show_warning(f"Point picker: {exc}")
            return
        affine = None
        if with_estimate:
            report = self.fit_report
            if report is None or not report.ok:
                show_warning(
                    "Point picker: there is no fit to preview yet."
                )
                return
            affine = report.affine
        picks = {
            "ids": [],
            "fixed": [],
            "moving": [],
        }
        for pair in self._complete_pairs().values():
            picks["ids"].append(pair.pair_id)
            picks["fixed"].append(pair.layer1_coords[:3])
            picks["moving"].append(pair.layer2_coords[:3])
        max_size = self.preview_size_spin.value()

        from napari.qt.threading import create_worker

        worker = create_worker(
            lambda: [self._load_preview_volume(j, max_size) for j in jobs]
        )
        worker.returned.connect(
            lambda volumes: self._show_preview(jobs, volumes, affine, picks)
        )
        worker.errored.connect(
            lambda exc: show_error(
                f"Point picker: the 3D preview failed: {exc}"
            )
        )
        worker.finished.connect(
            lambda: self._preview_workers.remove(worker)
            if worker in self._preview_workers
            else None
        )
        self._preview_workers.append(worker)
        show_info("Point picker: loading the 3D preview...")
        worker.start()

    def _show_preview(self, jobs, volumes, affine, picks):
        import napari

        title = (
            "3D preview: moving image with the current estimate"
            if affine is not None
            else "3D preview: raw data"
        )
        preview = napari.Viewer(
            title=title, ndisplay=3, show=self.preview_show_window
        )
        point_size = 1.0
        for job, (array, scale, translate) in zip(jobs, volumes, strict=True):
            kwargs = {
                "name": job["name"],
                "scale": scale,
                "translate": translate,
                "colormap": job["colormap"],
                "contrast_limits": job["contrast_limits"],
                "gamma": job["gamma"],
                "blending": "additive",
                "rendering": "mip",
            }
            if job["role"] == "moving" and affine is not None:
                kwargs["affine"] = affine
            preview.add_image(array, **kwargs)
            point_size = max(point_size, 4.0 * float(np.max(scale)))

        if picks["ids"]:
            moving = np.asarray(picks["moving"], dtype=float)
            if affine is not None:
                moving = apply_affine(affine, moving)
            for role, coords, job in (
                ("fixed", np.asarray(picks["fixed"], dtype=float), jobs[0]),
                ("moving", moving, jobs[1]),
            ):
                colour = job["colormap"].colors[-1]
                preview.add_points(
                    coords,
                    name=f"{role.capitalize()} picks",
                    size=point_size,
                    face_color="transparent",
                    border_color=colour,
                    border_width=0.15,
                    features={"pair": picks["ids"]},
                    text={"string": "{pair}", "color": "white", "size": 9},
                    blending="translucent_no_depth",
                )
        self._preview_viewers.append(preview)
        return preview


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
