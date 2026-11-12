"""Diagnostics for an affine fitted to matched point pairs.

The point picker recomputes these after every edit, so the user can judge a
fit (and spot a mismatched pair) before resampling the moving image.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

Estimator = Callable[[np.ndarray, np.ndarray], np.ndarray]

AXIS_NAMES_3D = ("Z", "Y", "X")

# A fit outside these bounds is flagged as implausible. Registration between
# views or cameras of one instrument is close to identity.
SCALE_RANGE = (0.8, 1.25)
MAX_ROTATION_DEG = 10.0
MAX_SHEAR = 0.2
# Picks are reported as nearly coplanar when their extent along the thinnest
# principal direction is below this fraction of the extent along the widest.
COPLANAR_RATIO = 0.1


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``(N, ndim)`` points through a homogeneous affine."""
    ndim = affine.shape[0] - 1
    pts = np.asarray(points, dtype=float)[:, :ndim]
    return pts @ affine[:ndim, :ndim].T + affine[:ndim, ndim]


@dataclass
class AffineSummary:
    """A 3D affine factored as rotation, per-axis scale, shear, translation."""

    scale: np.ndarray
    rotation_deg: float
    rotation_axis: np.ndarray
    shear: np.ndarray
    translation: np.ndarray

    def implausibilities(self) -> list[str]:
        issues = []
        lo, hi = SCALE_RANGE
        if np.any(self.scale < lo) or np.any(self.scale > hi):
            issues.append(
                f"scale {_fmt_vec(self.scale, 3)} is outside {lo} to {hi}"
            )
        if self.rotation_deg > MAX_ROTATION_DEG:
            issues.append(
                f"rotation {self.rotation_deg:.1f} deg exceeds "
                f"{MAX_ROTATION_DEG:g} deg"
            )
        if np.any(np.abs(self.shear) > MAX_SHEAR):
            issues.append(
                f"shear {_fmt_vec(self.shear, 3)} exceeds {MAX_SHEAR:g}"
            )
        return issues


def summarize_affine(affine: np.ndarray) -> AffineSummary | None:
    """Factor a 4x4 affine for display; ``None`` for other dimensions."""
    if affine.shape != (4, 4):
        return None
    from napari_orthogonal_views.point_picker_widget import (
        _decompose_linear_3d,
    )

    try:
        rotvec, scale, shear = _decompose_linear_3d(affine[:3, :3])
    except np.linalg.LinAlgError:
        return None
    angle = float(np.linalg.norm(rotvec))
    axis = rotvec / angle if angle > 0 else np.zeros(3)
    return AffineSummary(
        scale=scale,
        rotation_deg=float(np.degrees(angle)),
        rotation_axis=axis,
        shear=shear,
        translation=affine[:3, 3].copy(),
    )


@dataclass
class FitReport:
    """Everything the picker shows about the current fit.

    Residuals are in world units: the distance between a pair's fixed point
    and its moving point mapped through the fit.
    """

    n_complete: int
    n_used: int
    min_pairs: int
    affine: np.ndarray | None = None
    residual_norms: dict[int, float] = field(default_factory=dict)
    # Out-of-sample error per pair: for a pair used in the fit, its residual
    # under a fit made without it (leave-one-out); for an excluded pair, its
    # residual under the full fit.
    out_of_sample: dict[int, float] = field(default_factory=dict)
    rms: float | None = None
    max_pair: int | None = None
    loo_rms: float | None = None
    worst_loo_pair: int | None = None
    stability: float | None = None
    stability_sigma: float | None = None
    coverage: np.ndarray | None = None
    summary: AffineSummary | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.affine is not None

    @property
    def pairs_needed(self) -> int:
        return max(0, self.min_pairs - self.n_used)


def evaluate_fit(
    estimator: Estimator,
    fixed: Mapping[int, Sequence[float]],
    moving: Mapping[int, Sequence[float]],
    used_ids: Sequence[int],
    *,
    min_pairs: int | None = None,
    volume_corners: np.ndarray | None = None,
    pick_sigma: float = 0.5,
    n_jitter: int = 8,
    seed: int = 0,
) -> FitReport:
    """Fit ``moving -> fixed`` on ``used_ids`` and diagnose the result.

    Parameters
    ----------
    estimator
        ``(source, target) -> affine``, called with moving points as source.
    fixed, moving
        World coordinates of every complete pair, keyed by pair id.
    used_ids
        Pairs to fit on. Pairs not listed still get a residual.
    min_pairs
        Fewest pairs the estimator accepts. Defaults to ``ndim + 1``.
    volume_corners
        ``(K, ndim)`` world positions spanning the moving image. Enables
        the coverage figure and the stability check.
    pick_sigma
        Standard deviation of the simulated pick noise, in world units.
    n_jitter
        Number of noisy refits for the stability check (0 disables it).
    seed
        Seed for the simulated noise, so the figure does not flicker
        between identical edits.
    """
    ids = list(fixed)
    used = [i for i in used_ids if i in fixed]
    ndim = len(next(iter(fixed.values()))) if fixed else 3
    if min_pairs is None:
        min_pairs = ndim + 1
    report = FitReport(
        n_complete=len(ids), n_used=len(used), min_pairs=min_pairs
    )
    if len(used) < min_pairs:
        return report

    src = np.array([moving[i] for i in used], dtype=float)[:, :ndim]
    tgt = np.array([fixed[i] for i in used], dtype=float)[:, :ndim]

    try:
        affine = np.asarray(estimator(src, tgt), dtype=float)
        if not np.all(np.isfinite(affine)):
            raise ValueError("estimator returned non-finite values")
        np.linalg.inv(affine[:ndim, :ndim])
    except Exception as exc:  # noqa: BLE001 (shown to the user)
        report.error = f"{type(exc).__name__}: {exc}"
        return report
    report.affine = affine

    all_src = np.array([moving[i] for i in ids], dtype=float)[:, :ndim]
    all_tgt = np.array([fixed[i] for i in ids], dtype=float)[:, :ndim]
    norms = np.linalg.norm(apply_affine(affine, all_src) - all_tgt, axis=1)
    report.residual_norms = dict(zip(ids, norms.tolist(), strict=True))

    used_norms = np.array([report.residual_norms[i] for i in used])
    report.rms = float(np.sqrt(np.mean(used_norms**2)))
    report.max_pair = used[int(np.argmax(used_norms))]

    _leave_one_out(report, estimator, fixed, moving, used, ndim)
    corners = (
        None if volume_corners is None else np.asarray(volume_corners, float)
    )
    if corners is not None and corners.ndim == 2 and corners.shape[1] >= ndim:
        corners = corners[:, :ndim]
        _coverage(report, src, corners)
        if n_jitter > 0 and pick_sigma > 0:
            _stability(
                report,
                estimator,
                src,
                tgt,
                corners,
                pick_sigma,
                n_jitter,
                seed,
            )
    _coplanarity(report, src)

    report.summary = summarize_affine(affine)
    if report.summary is not None:
        for issue in report.summary.implausibilities():
            report.warnings.append(
                f"The fit looks implausible for a near-identity alignment: "
                f"{issue}. Check for a mismatched pair."
            )
    return report


def _leave_one_out(report, estimator, fixed, moving, used, ndim) -> None:
    for pair_id, norm in report.residual_norms.items():
        if pair_id not in used:
            report.out_of_sample[pair_id] = norm
    if len(used) - 1 < report.min_pairs:
        return
    for pair_id in used:
        others = [i for i in used if i != pair_id]
        src = np.array([moving[i] for i in others], dtype=float)[:, :ndim]
        tgt = np.array([fixed[i] for i in others], dtype=float)[:, :ndim]
        try:
            loo_affine = estimator(src, tgt)
        except Exception:  # noqa: BLE001 (skip a rejected refit)
            continue
        pred = apply_affine(
            loo_affine, np.array([moving[pair_id]], dtype=float)
        )[0]
        report.out_of_sample[pair_id] = float(
            np.linalg.norm(pred - np.asarray(fixed[pair_id])[:ndim])
        )
    loo = [report.out_of_sample[i] for i in used if i in report.out_of_sample]
    if loo:
        report.loo_rms = float(np.sqrt(np.mean(np.square(loo))))
        report.worst_loo_pair = max(
            (i for i in used if i in report.out_of_sample),
            key=report.out_of_sample.__getitem__,
        )


def _coverage(report, src, corners) -> None:
    extent = np.ptp(corners, axis=0)
    spread = np.ptp(src, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        report.coverage = np.where(extent > 0, spread / extent, 0.0)


def _stability(
    report, estimator, src, tgt, corners, sigma, n_jitter, seed
) -> None:
    """Refit on noisy copies of the picks and measure how far corners move."""
    rng = np.random.default_rng(seed)
    reference = apply_affine(report.affine, corners)
    sq_disp = []
    for _ in range(n_jitter):
        try:
            noisy = estimator(
                src + rng.normal(0.0, sigma, src.shape),
                tgt + rng.normal(0.0, sigma, tgt.shape),
            )
        except Exception:  # noqa: BLE001 (skip a rejected refit)
            continue
        disp = apply_affine(noisy, corners) - reference
        sq_disp.append(np.sum(disp**2, axis=1))
    if sq_disp:
        per_corner = np.sqrt(np.mean(sq_disp, axis=0))
        report.stability = float(per_corner.max())
        report.stability_sigma = sigma


def _coplanarity(report, src) -> None:
    ndim = src.shape[1]
    if ndim != 3 or len(src) < 3:
        return
    centred = src - src.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    extents = np.ptp(centred @ vt.T, axis=0)
    if extents[0] <= 0 or extents[-1] >= COPLANAR_RATIO * extents[0]:
        return
    thin = np.abs(vt[-1])
    axis = int(np.argmax(thin))
    direction = (
        f"mostly along {AXIS_NAMES_3D[axis]}"
        if thin[axis] > 0.9
        else f"direction (z, y, x) = {_fmt_vec(vt[-1], 2)}"
    )
    report.warnings.append(
        f"Picks are nearly coplanar: they span {extents[-1]:.1f} along "
        f"their thinnest direction ({direction}) versus {extents[0]:.1f} "
        "along the widest. Terms that depend on that direction are poorly "
        "constrained unless the estimator holds them fixed; check the "
        "stability figure."
    )


def _fmt_vec(values, decimals: int) -> str:
    return "(" + ", ".join(f"{v:.{decimals}f}" for v in values) + ")"
