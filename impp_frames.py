"""How impp places a stored view, and how a bead fit turns into calibration matrices.

impp (``royerlab/impp``, ``Project.view``) stores each deskewed view in its own grid. A raw voxel ``r`` of the view lands
at stored voxel ``M @ after @ deskew @ camera @ pre @ r``:

- ``pre``: the camera's pre-deskew flips (``camera_axes_to_flip_pre_deskew``)
- ``camera``: its (Y, X) camera alignment, embedded in 4x4
- ``deskew``: the view's matrix as impp loads it, which is the file's matrix rescaled to the plate's frame step
  (``F @ D_file @ F^-1``, ``F = diag(1/r, 1, 1, 1)`` with ``r`` the plate's step over ``frame_step_um``) with the top
  arm's y shift taken out of views 2 and 3 (``Sh``, impp's ``top_y_shift``)
- ``after``: the camera's post-deskew flips
- ``M``: the y mirror impp stores views 2 and 3 with

The picker shows A at its stored voxels and B shifted by ``o``, its origin in A's voxels as the project's alignment
places it, so world coordinates are A's stored voxels. In the picker's slot model each view's raw-to-world chain is
``POST @ A @ PREV`` with ``A`` the file's matrix::

    POST = T(o) @ M @ after @ Sh @ F        PREV = F^-1 @ camera @ pre

and a world-space correction ``W`` (B's beads onto A's) gives ``A_new = inv(POST) @ W @ POST @ A_old``, a matrix in the
file's own frame, ready for ``pick_projects.py assemble``.

A camera alignment is (Y, X) on the raw frame, which no 3D world correction keeps, so a camera fit maps each pick back to
B's raw frame and solves there: ``camera_new @ pre @ source = camera @ pre @ target``. Two cameras expose a frame
together, so the picks' frame (z) difference in raw is pick error, reported but not fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def identity4() -> np.ndarray:
    return np.eye(4)


def translation(offset_zyx) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(offset_zyx, dtype=float)
    return m


def flip(shape_zyx, axes) -> np.ndarray:
    """The 4x4 reversing *axes* of a grid of *shape_zyx* voxels (voxel i to n - 1 - i)."""
    m = np.eye(4)
    for axis in axes:
        m[int(axis), int(axis)] = -1.0
        m[int(axis), 3] = shape_zyx[int(axis)] - 1.0
    return m


def embed_camera(matrix_3x3) -> np.ndarray:
    """A (Y, X) 3x3 affine as the 4x4 that leaves z alone."""
    a = np.asarray(matrix_3x3, dtype=float)
    m = np.eye(4)
    m[1:3, 1:3], m[1:3, 3] = a[:2, :2], a[:2, 2]
    return m


def camera_3x3(matrix_4x4) -> np.ndarray:
    """The (Y, X) 3x3 a 4x4 camera alignment embeds."""
    m = np.asarray(matrix_4x4, dtype=float)
    return np.array([[m[1, 1], m[1, 2], m[1, 3]], [m[2, 1], m[2, 2], m[2, 3]], [0.0, 0.0, 1.0]])


def apply(matrix, points) -> np.ndarray:
    """(N, d) *points* through a homogeneous (d + 1)-square *matrix*."""
    p = np.asarray(points, dtype=float)
    return (np.c_[p, np.ones(len(p))] @ np.asarray(matrix, dtype=float).T)[:, :-1]


@dataclass(frozen=True)
class ViewFrame:
    """One stored view as impp made it: the factors above, the file's own matrix, the stored *shape*, whether it is
    stored *mirrored* in y, and *origin*, where the picker draws it (A's stored voxels)."""

    pre: np.ndarray
    camera: np.ndarray
    deskew_loaded: np.ndarray
    deskew_file: np.ndarray
    after: np.ndarray
    frame_ratio: float
    top_shift: float
    view: int
    shape: tuple
    mirrored: bool
    origin: np.ndarray

    def mirror(self) -> np.ndarray:
        return flip(self.shape, (1,)) if self.mirrored else np.eye(4)

    def frame_step(self) -> np.ndarray:
        """F: the file's frame step to the plate's."""
        return np.diag([1.0 / self.frame_ratio, 1.0, 1.0, 1.0])

    def top(self) -> np.ndarray:
        """Sh: the top arm's y shift impp takes out at load (identity on views 0 and 1)."""
        return translation((0.0, -self.top_shift, 0.0)) if self.view in (2, 3) else np.eye(4)

    def post(self) -> np.ndarray:
        return translation(self.origin) @ self.mirror() @ self.after @ self.top() @ self.frame_step()

    def prev(self) -> np.ndarray:
        return np.linalg.inv(self.frame_step()) @ self.camera @ self.pre

    def raw_to_view(self) -> np.ndarray:
        """Raw voxel to grid voxel as impp applies it (what the stored view's geometry records)."""
        return self.after @ self.deskew_loaded @ self.camera @ self.pre

    def check(self, recorded_raw_to_view) -> None:
        """Refuse a frame that does not reproduce what impp recorded, or whose file matrix does not load to the matrix
        impp applied: a matrix solved in the wrong frame would be written silently otherwise."""
        if not np.allclose(self.raw_to_view(), recorded_raw_to_view, atol=1e-6):
            raise SystemExit(f"view {self.view}: the factors read from impp do not reproduce the stored view's "
                             "raw_to_view (deskewed with other matrices?); deskew the view again")
        loaded = self.top() @ self.frame_step() @ self.deskew_file @ np.linalg.inv(self.frame_step())
        if not np.allclose(loaded, self.deskew_loaded, atol=1e-6):
            raise SystemExit(f"view {self.view}: the pinned file's matrix does not load to the one impp applied "
                             "(frame step or top shift not as modelled); refusing to solve")


def solve_slot(frame: ViewFrame, world_correction) -> np.ndarray:
    """``A_new = inv(POST) @ W @ POST @ A_old``: the file matrix that moves this view by *world_correction*."""
    post = frame.post()
    return np.linalg.inv(post) @ np.asarray(world_correction, float) @ post @ frame.deskew_file


def matrix_square_root(matrix) -> np.ndarray:
    """The real affine whose square is *matrix* (the shared-space split)."""
    from scipy.linalg import sqrtm

    root = np.real_if_close(sqrtm(np.asarray(matrix, float)), tol=1000)
    if np.iscomplexobj(root):
        raise RuntimeError("the estimated affine has no real square root; use --output-mode slot")
    root = np.asarray(root, float)
    root[-1, :] = 0.0
    root[-1, -1] = 1.0
    return root


@dataclass(frozen=True)
class CameraFit:
    """A camera fit: the new (Y, X) 3x3 *matrix*, its *residuals* (N, 2) in raw pixels, and *frame_rms*, the RMS of
    the picks' raw frame (z) difference, which a camera alignment cannot take up."""

    matrix: np.ndarray
    residuals: np.ndarray
    frame_rms: float

    def rms(self) -> tuple:
        return tuple(float(v) for v in np.sqrt((self.residuals ** 2).mean(axis=0)))


def fit_affine_2d(source, target) -> np.ndarray:
    """Least-squares (Y, X) affine, 3x3, taking *source* (N, 2) to *target*."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 3 or np.linalg.matrix_rank(np.c_[source, np.ones(len(source))]) < 3:
        raise ValueError("a camera fit needs at least 3 picks not on one line (8 to 10 across the field is better)")
    solution, *_ = np.linalg.lstsq(np.c_[source, np.ones(len(source))], target, rcond=None)
    m = np.eye(3)
    m[:2, :] = solution.T
    return m


def camera_fit(frame_b: ViewFrame, world_a, world_b) -> CameraFit:
    """The camera alignment that takes B's beads (*world_b*, world voxels) onto A's (*world_a*), solved in B's raw."""
    to_raw = np.linalg.inv(frame_b.raw_to_view()) @ frame_b.mirror() @ translation(-np.asarray(frame_b.origin))
    source = apply(frame_b.pre, apply(to_raw, world_b))
    target = apply(frame_b.camera @ frame_b.pre, apply(to_raw, world_a))
    matrix = fit_affine_2d(source[:, 1:], target[:, 1:])
    residuals = apply(matrix, source[:, 1:]) - target[:, 1:]
    frame_rms = float(np.sqrt(((source[:, 0] - target[:, 0]) ** 2).mean()))
    return CameraFit(matrix, residuals, frame_rms)


def camera_key(transforms: dict, camera_id: int, view: int, applied_3x3) -> str:
    """The key under which the file holds the camera alignment impp applied to this camera and view. impp may key a
    file's cameras by dataset order rather than ``camera_name_to_id``, so the key is found by value;
    ``camera_<id>`` wins a tie."""
    keys = [k for k, per_view in (transforms or {}).items() if isinstance(per_view, dict)
            and f"view_{view}" in per_view
            and np.allclose(np.asarray(per_view[f"view_{view}"], float), applied_3x3, atol=1e-6)]
    if f"camera_{camera_id}" in keys:
        return f"camera_{camera_id}"
    if len(keys) == 1:
        return keys[0]
    raise SystemExit(f"cannot tell which camera_alignment key of the pinned file impp applied to camera {camera_id}, "
                     f"view {view} (keys holding that matrix: {keys or 'none'})")
