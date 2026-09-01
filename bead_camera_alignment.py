"""Fit a DaXi inter-camera alignment from a bead stack and emit the YAML block.

This replaces Step 1 of the manual (GUI point-picking per view) for datasets
that contain beads. It finds bead correspondences automatically, fits the same
model the point picker's CAMERA mode fits, and prints a ready-to-paste
``camera_alignment`` block covering every view in one run.

The model is an in-plane (Y, X) affine with Z left untouched. The fit maps
moving-zarr voxels onto fixed-zarr voxels, in the frame the pipeline applies
alignment in, that is, after the per-camera pre-deskew flips.

Which camera the pipeline applies a ``camera_alignment`` matrix to is decided
by enumeration order, alphabetical by serial, not by ``camera_name_to_id``
(that mapping governs only flips and deskew transforms). On DaXi A that makes
camera_0 = SN302483 (561 nm) and camera_1 = SN304410 (488 nm). So the printed
block stores the INVERSE of the fit under the fixed zarr's camera key, and
makes the moving zarr's camera the reference. Pass the enumeration ids as
``--fixed-camera-id`` / ``--moving-camera-id``; the block is then paste-ready.

Correspondences come from bead centroids in max projections over thin Z slabs.
Projecting thins the data enough to make peak finding cheap while keeping
crowding low, and a slab keeps the subpixel centroid of a bead intact because
the fitted model does not touch Z.

Example, DaXi A, fitting 561 nm onto the X-flipped 488 nm frame:

    uv run python3 bead_camera_alignment.py \
      --fixed-zarr-path  $ZARR/SN304410/488nm/pos0 --fixed-camera-id 1 \
      --moving-zarr-path $ZARR/SN302483/561nm/pos0 --moving-camera-id 0 \
      --flip-fixed-x \
      --channels 0,1,2,3

``--flip-fixed-x`` and friends must reproduce the microscope's
``camera_axes_to_flip_pre_deskew``, which is keyed by ``camera_name_to_id``
(SN304410 -> 0 -> flip X on DaXi A): axis 1 is Y, axis 2 is X, and a camera
listed there needs the matching flip flag on its side of the fit. Getting this
wrong shows up immediately as a fit that will not converge below a few pixels.

Check the reported RMS before trusting a matrix. Under about 2 px at full
resolution is a good fit for this metric; a view that lands well above that,
or that yields only a handful of pairs, usually means a wrong flip flag, a
channel that holds no beads, or a genuinely misaligned camera.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation


@dataclass
class Fit:
    """One view's fitted alignment and its quality."""

    view_id: int
    matrix: np.ndarray  # 3x3, rows (Y, X), moving -> fixed
    rms: float
    max_residual: float
    n_pairs: int


def load_volume(
    path: Path, channel: int, timepoint: int, level: str, flip_axes: tuple[int, ...]
) -> np.ndarray:
    """Read one (Z, Y, X) volume and apply pre-deskew flips."""
    group = zarr.open(str(path), mode="r")
    array = group[level] if level in group else group
    if array.ndim != 5:
        raise SystemExit(
            f"{path}/{level} has shape {array.shape}; expected 5D (T, C, Z, Y, X)"
        )
    volume = np.asarray(array[timepoint, channel])
    for axis in flip_axes:
        volume = np.flip(volume, axis=axis)
    return np.ascontiguousarray(volume)


def detect_beads(image: np.ndarray, min_distance: int, sigma_cut: float) -> np.ndarray:
    """Subpixel (row, col) bead centroids in a 2D projection."""
    background = float(np.median(image))
    mad = float(np.median(np.abs(image - background)))
    peaks = peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=background + sigma_cut * max(mad, 1.0),
        exclude_border=min_distance + 1,
    )
    if peaks.size == 0:
        return np.zeros((0, 2))

    half = 2
    offsets = np.arange(-half, half + 1)
    grid_row, grid_col = np.meshgrid(offsets, offsets, indexing="ij")
    centroids = []
    for row, col in peaks:
        window = image[row - half : row + half + 1, col - half : col + half + 1]
        window = np.clip(window.astype(float) - background, 0, None)
        total = window.sum()
        if total <= 0:
            continue
        centroids.append(
            (
                row + float((window * grid_row).sum() / total),
                col + float((window * grid_col).sum() / total),
            )
        )
    return np.asarray(centroids)


def match_points(
    moving: np.ndarray,
    fixed: np.ndarray,
    predicted: np.ndarray,
    radius: float,
    uniqueness: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Match beads by nearest neighbour, keeping only unambiguous pairs.

    A pair survives only if the two beads are each other's nearest neighbour
    and the runner-up is at least ``uniqueness`` times further away, which
    keeps dense bead fields from contributing plausible but wrong pairs.
    """
    if len(moving) == 0 or len(fixed) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    fixed_tree, moving_tree = cKDTree(fixed), cKDTree(predicted)
    distances, indices = fixed_tree.query(predicted, k=min(2, len(fixed)))
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]

    keep_moving, keep_fixed = [], []
    for index, (distance_row, index_row) in enumerate(zip(distances, indices)):
        if distance_row[0] > radius:
            continue
        if len(distance_row) > 1 and distance_row[1] < uniqueness * max(distance_row[0], 1e-6):
            continue
        target = int(index_row[0])
        if int(moving_tree.query(fixed[target], k=1)[1]) != index:
            continue
        keep_moving.append(moving[index])
        keep_fixed.append(fixed[target])
    return np.asarray(keep_moving), np.asarray(keep_fixed)


def fit_affine(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Least-squares (Y, X) affine with fixed ~= matrix @ moving."""
    design = np.column_stack([moving, np.ones(len(moving))])
    solution, *_ = np.linalg.lstsq(design, fixed, rcond=None)
    matrix = np.eye(3)
    matrix[:2, :2] = solution[:2].T
    matrix[:2, 2] = solution[2]
    return matrix


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def residuals(matrix: np.ndarray, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_affine(matrix, moving) - fixed, axis=1)


def collect_pairs(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    coarse: np.ndarray,
    *,
    slab: int,
    min_distance: int,
    sigma_cut: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = fixed_volume.shape[0]
    moving_points, fixed_points = [], []
    for start in range(0, depth, slab):
        stop = min(start + slab, depth)
        fixed_beads = detect_beads(
            fixed_volume[start:stop].max(axis=0), min_distance, sigma_cut
        )
        moving_beads = detect_beads(
            moving_volume[start:stop].max(axis=0), min_distance, sigma_cut
        )
        predicted = apply_affine(coarse, moving_beads) if len(moving_beads) else moving_beads
        matched_moving, matched_fixed = match_points(
            moving_beads, fixed_beads, predicted, radius=radius
        )
        if len(matched_moving):
            moving_points.append(matched_moving)
            fixed_points.append(matched_fixed)
    if not moving_points:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(moving_points), np.concatenate(fixed_points)


def estimate(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    view_id: int,
    *,
    slab: int,
    min_distance: int,
    sigma_cut: float,
    radius: float,
) -> Fit:
    shift, _, _ = phase_cross_correlation(
        fixed_volume.max(axis=0).astype(np.float32),
        moving_volume.max(axis=0).astype(np.float32),
        upsample_factor=4,
    )
    # A dense bead field can hand phase correlation a wrong peak, so search from
    # both the correlation shift and no shift, and keep whichever matches more
    # beads. A wrong seed matches very few, so the count separates them cleanly.
    best_pairs: tuple[np.ndarray, np.ndarray] = (np.zeros((0, 2)), np.zeros((0, 2)))
    best_seed = None
    for name, candidate in (("phase-correlation", shift), ("zero", np.zeros(2))):
        coarse = np.eye(3)
        coarse[:2, 2] = candidate
        pairs = collect_pairs(
            fixed_volume,
            moving_volume,
            coarse,
            slab=slab,
            min_distance=min_distance,
            sigma_cut=sigma_cut,
            radius=radius,
        )
        print(
            f"  seed {name} (dY, dX)={np.round(candidate, 2)}: {len(pairs[0])} pairs",
            flush=True,
        )
        if len(pairs[0]) > len(best_pairs[0]):
            best_pairs, best_seed = pairs, name

    moving_points, fixed_points = best_pairs
    if len(moving_points) < 20:
        raise SystemExit(
            f"view {view_id}: only {len(moving_points)} bead pairs matched. Check the "
            "flip flags, the channel index, and that this channel contains beads."
        )
    print(f"  seed {best_seed} wins with {len(moving_points)} pairs", flush=True)

    def trim_and_fit(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = fit_affine(moving, fixed)
        for cut in (radius / 2, radius / 4, radius / 8):
            keep = residuals(matrix, moving, fixed) < cut
            if keep.sum() < 20:
                break
            moving, fixed = moving[keep], fixed[keep]
            matrix = fit_affine(moving, fixed)
            error = residuals(matrix, moving, fixed)
            print(
                f"  outlier cut {cut:>5.2f} px: {len(moving):5d} pairs, "
                f"RMS={np.sqrt((error**2).mean()):.3f} max={error.max():.3f}",
                flush=True,
            )
        return matrix, moving, fixed

    matrix, moving_points, fixed_points = trim_and_fit(moving_points, fixed_points)

    # Re-match using the fitted model rather than the coarse seed. This recovers
    # correspondences the seed was too far off to find, which widens the field
    # the fit is constrained over and makes the result independent of which seed
    # won above.
    rematched = collect_pairs(
        fixed_volume,
        moving_volume,
        matrix,
        slab=slab,
        min_distance=min_distance,
        sigma_cut=sigma_cut,
        radius=max(radius / 2, 4.0),
    )
    if len(rematched[0]) > len(moving_points):
        print(f"  re-match with fitted model: {len(rematched[0])} pairs", flush=True)
        matrix, moving_points, fixed_points = trim_and_fit(*rematched)

    error = residuals(matrix, moving_points, fixed_points)
    return Fit(
        view_id=view_id,
        matrix=matrix,
        rms=float(np.sqrt((error**2).mean())),
        max_residual=float(error.max()),
        n_pairs=int(len(moving_points)),
    )


def stored_matrix(fit: Fit) -> np.ndarray:
    """What goes in the YAML: the inverse of the fit.

    The fit maps moving-zarr voxels onto fixed-zarr voxels, but the pipeline
    applies the fixed zarr's camera key to the fixed camera's data, so that
    camera must carry the inverse map. The moving camera stays the reference.
    """
    inverse = np.linalg.inv(fit.matrix)
    inverse[2] = [0.0, 0.0, 1.0]
    return inverse


def format_yaml_block(fits: list[Fit], fixed_camera_id: int, moving_camera_id: int) -> str:
    lines = [
        "camera_alignment:",
        f"  reference_camera: {moving_camera_id}",
        "  transforms:",
        f"    camera_{moving_camera_id}:",
        f"      view_{fits[0].view_id}: &identity",
        "        - [1.0, 0.0, 0.0]",
        "        - [0.0, 1.0, 0.0]",
        "        - [0.0, 0.0, 1.0]",
    ]
    lines += [f"      view_{fit.view_id}: *identity" for fit in fits[1:]]
    lines.append(f"    camera_{fixed_camera_id}:")
    for fit in fits:
        lines.append(
            f"      view_{fit.view_id}:"
            f"    # inverse of fit, RMS {fit.rms:.2f} px, {fit.n_pairs} bead pairs"
        )
        for row in stored_matrix(fit):
            lines.append("        - [" + ", ".join(f"{value:.10f}" for value in row) + "]")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fixed-zarr-path",
        type=Path,
        required=True,
        help="Reference camera position group, e.g. .../SN304410/488nm/pos0",
    )
    parser.add_argument(
        "--moving-zarr-path",
        type=Path,
        required=True,
        help="Moving camera position group, e.g. .../SN302483/561nm/pos0",
    )
    parser.add_argument(
        "--channels",
        default="0,1,2,3",
        help="Channel indices to fit. Both cameras are read at the same index.",
    )
    parser.add_argument(
        "--view-ids",
        default=None,
        help=(
            "View id per channel, same order as --channels. Defaults to the "
            "channel indices, which is correct when channels are ordered "
            "bottom-A, bottom-B, top-A, top-B."
        ),
    )
    parser.add_argument(
        "--fixed-camera-id",
        type=int,
        required=True,
        help=(
            "Pipeline camera id of the fixed zarr, in enumeration order "
            "(alphabetical by serial; SN304410 is 1 on DaXi A). Its key "
            "receives the inverse of the fit."
        ),
    )
    parser.add_argument(
        "--moving-camera-id",
        type=int,
        required=True,
        help=(
            "Pipeline camera id of the moving zarr, in enumeration order "
            "(SN302483 is 0 on DaXi A). It becomes the reference camera."
        ),
    )
    parser.add_argument("--timepoint", type=int, default=0)
    parser.add_argument("--level", default="0", help="Multiscale level to read.")
    parser.add_argument(
        "--flip-fixed-y", action="store_true", help="Reference camera flips Y pre-deskew."
    )
    parser.add_argument(
        "--flip-fixed-x", action="store_true", help="Reference camera flips X pre-deskew."
    )
    parser.add_argument(
        "--flip-moving-y", action="store_true", help="Moving camera flips Y pre-deskew."
    )
    parser.add_argument(
        "--flip-moving-x", action="store_true", help="Moving camera flips X pre-deskew."
    )
    parser.add_argument(
        "--slab",
        type=int,
        default=50,
        help="Z slab thickness projected before bead finding.",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=6,
        help="Minimum separation between detected beads, in pixels.",
    )
    parser.add_argument(
        "--sigma-cut",
        type=float,
        default=25.0,
        help="Bead threshold in robust deviations above the background median.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=16.0,
        help="Largest camera offset to search for, in pixels.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the fits, residuals and pair counts here.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channels = [int(value) for value in args.channels.split(",")]
    view_ids = (
        [int(value) for value in args.view_ids.split(",")] if args.view_ids else channels
    )
    if len(view_ids) != len(channels):
        raise SystemExit("--view-ids must have one entry per --channels entry")

    fixed_flips = tuple(
        axis for axis, flip in ((1, args.flip_fixed_y), (2, args.flip_fixed_x)) if flip
    )
    moving_flips = tuple(
        axis for axis, flip in ((1, args.flip_moving_y), (2, args.flip_moving_x)) if flip
    )
    print(f"fixed  flips: axes {fixed_flips or 'none'}")
    print(f"moving flips: axes {moving_flips or 'none'}")

    fits = []
    for channel, view_id in zip(channels, view_ids):
        print(f"\nchannel {channel} -> view {view_id}", flush=True)
        fixed_volume = load_volume(
            args.fixed_zarr_path, channel, args.timepoint, args.level, fixed_flips
        )
        moving_volume = load_volume(
            args.moving_zarr_path, channel, args.timepoint, args.level, moving_flips
        )
        fit = estimate(
            fixed_volume,
            moving_volume,
            view_id,
            slab=args.slab,
            min_distance=args.min_distance,
            sigma_cut=args.sigma_cut,
            radius=args.radius,
        )
        del fixed_volume, moving_volume
        print(
            f"  view {view_id}: RMS={fit.rms:.3f} px  max={fit.max_residual:.3f} px  "
            f"n={fit.n_pairs}",
            flush=True,
        )
        fits.append(fit)

    print("\n" + "=" * 72)
    print("Paste into the microscope's transforms YAML (already inverted for the")
    print(f"pipeline, camera_{args.fixed_camera_id} receives it):")
    print("=" * 72)
    print(format_yaml_block(fits, args.fixed_camera_id, args.moving_camera_id))

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    str(fit.view_id): {
                        "matrix": fit.matrix.tolist(),
                        "stored_inverse": stored_matrix(fit).tolist(),
                        "rms": fit.rms,
                        "max_residual": fit.max_residual,
                        "n_pairs": fit.n_pairs,
                    }
                    for fit in fits
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
