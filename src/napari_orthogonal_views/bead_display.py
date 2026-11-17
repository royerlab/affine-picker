"""Display filter that makes fluorescent beads stand out from their background.

Bead volumes have a smooth background (camera offset, scattered light, and a
gradient along the light sheet) of a few hundred counts, while the beads
themselves span roughly 1,000 to 65,535 counts. Shown with fixed contrast
limits, most beads are a few percent of the display range; lowering gamma
brings them up but lifts the background with them. :func:`enhance_beads`
removes the background so the floor is black, and returns contrast limits
taken from the beads:

1. Background: a low percentile of each block of voxels (beads are sparse
   peaks, so they barely move it), linearly interpolated back to every
   voxel.
2. Noise floor: the typical background level (median of the subtracted
   volume) plus a multiple of its noise (median absolute deviation) is
   subtracted and the result clipped at zero, so most pixel noise is black.
3. Despeckle: a voxel left above zero with fewer than 8 of its 26
   neighbours also above zero is set to zero. Noise that crosses the floor
   is mostly isolated voxels; a bead is a connected blob, and its apex keeps
   its full sharpness (a median filter would blur it).

Non-finite input is common in stored float16 views: NaN (padding) is shown as
background and inf (overflow on bright beads) as saturated.

The steps subtract a smooth field, clip, and remove isolated voxels, so bead
positions are unchanged; the filter changes only what is displayed.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi

# Background block in full-resolution voxels (z, y, x). Larger than a bead
# and its PSF tails, small enough to follow the gradient along y.
BACKGROUND_BLOCK = (8, 32, 32)
BACKGROUND_PERCENTILE = 10.0
NOISE_FLOOR = 1.0  # noise standard deviations above the background median
DESPECKLE_NEIGHBOURS = 8  # of 26; 0 turns despeckling off
UPPER_PERCENTILE = 99.95  # upper contrast limit
DISPLAY_GAMMA = 0.5


def _interp_indices(
    n_out: int, n_in: int, block: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbouring block centres and weight for linear interpolation.

    Returns ``(i0, i1, t)`` such that the value at output index ``j`` is
    ``(1 - t[j]) * v[i0[j]] + t[j] * v[i1[j]]``. Past the outermost
    centres the line through the last two is extended, so a gradient is
    followed to the volume's edge.
    """
    pos = np.arange(n_out, dtype=float)
    if n_in == 1:
        zeros = np.zeros(n_out, dtype=np.intp)
        return zeros, zeros, np.zeros(n_out, dtype=np.float32)
    centres = (np.arange(n_in) + 0.5) * block - 0.5
    i0 = np.clip(np.searchsorted(centres, pos, side="right") - 1, 0, n_in - 2)
    i1 = i0 + 1
    t = ((pos - centres[i0]) / (centres[i1] - centres[i0])).astype(np.float32)
    return i0, i1, t


def _block_background(
    volume: np.ndarray, block: tuple[int, int, int], percentile: float
) -> np.ndarray:
    """Low percentile of each block: shape ceil(volume / block).

    Every second voxel in y and x is used when the block is at least 4 wide
    there, which leaves thousands of samples per block at full resolution.
    """
    bz, by, bx = block
    nz, ny, nx = (-(-s // b) for s, b in zip(volume.shape, block, strict=True))
    sy = 2 if by >= 4 and by % 2 == 0 else 1
    sx = 2 if bx >= 4 and bx % 2 == 0 else 1
    grid = np.empty((nz, ny, nx), dtype=np.float32)

    def one(k: int) -> None:
        slab = volume[k * bz : (k + 1) * bz]
        pad = ((0, 0), (0, ny * by - slab.shape[1]), (0, nx * bx - slab.shape[2]))
        slab = np.pad(slab, pad, mode="edge")[:, ::sy, ::sx]
        blocks = slab.reshape(slab.shape[0], ny, by // sy, nx, bx // sx)
        blocks = blocks.transpose(1, 3, 0, 2, 4).reshape(ny, nx, -1)
        kth = int(round(percentile / 100 * (blocks.shape[-1] - 1)))
        grid[k] = np.partition(blocks, kth, axis=-1)[..., kth]

    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
        list(pool.map(one, range(nz)))
    bad = ~np.isfinite(grid)
    if bad.all():
        return np.zeros_like(grid)
    if bad.any():  # blocks of mostly NaN (padding): take the nearest block's value
        nearest = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
        grid = grid[tuple(nearest)]
    return grid


def _isolated(above: np.ndarray, min_neighbours: int) -> np.ndarray:
    """Voxels of the inner planes of *above* with too few set neighbours.

    *above* is a boolean (z, y, x) slab whose first and last planes are
    context only; the result covers planes 1 to -1. Neighbours are the 26
    voxels around each one; outside the volume counts as not set.
    """
    m = np.pad(above, ((0, 0), (1, 1), (1, 1))).view(np.uint8)
    s = m[:, :, :-2] + m[:, :, 1:-1] + m[:, :, 2:]
    s = s[:, :-2] + s[:, 1:-1] + s[:, 2:]
    s = s[:-2] + s[1:-1] + s[2:]
    centre = m[1:-1, 1:-1, 1:-1]
    return (centre == 1) & (s - centre < min_neighbours)


def enhance_beads(
    volume: np.ndarray,
    *,
    level_factor: float | tuple[float, float, float] = 1,
    block: tuple[int, int, int] = BACKGROUND_BLOCK,
    percentile: float = BACKGROUND_PERCENTILE,
    noise_floor: float = NOISE_FLOOR,
    despeckle_neighbours: int = DESPECKLE_NEIGHBOURS,
    slab: int = 32,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Background-subtract a bead volume for display.

    Parameters
    ----------
    volume
        3D (z, y, x) volume, any numeric dtype.
    level_factor
        Downsampling factor of *volume* relative to full resolution, one
        value or one per (z, y, x) axis, so
        *block* (given in full-resolution voxels) covers the same region at
        every pyramid level.
    block, percentile
        Background block size (full-resolution voxels) and the percentile
        of each block taken as its background.
    noise_floor
        Noise standard deviations above the background median that are
        subtracted after the background. About 84% of background voxels
        are zero at 1.0 and 98% at 2.0, for Gaussian noise.
    despeckle_neighbours
        A voxel above the floor with fewer of its 26 neighbours above the
        floor is set to zero. 0 turns despeckling off.
    slab
        Planes processed at a time; bounds the temporary memory.

    Returns
    -------
    enhanced : np.ndarray
        uint16 volume of the same shape, zero at the background.
    contrast_limits : tuple of float
        ``(0, upper)``, with *upper* the 99.95th percentile of the
        enhanced volume.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    factors = np.broadcast_to(np.asarray(level_factor, dtype=float), (3,))
    level_block = tuple(
        max(2, round(b / f)) for b, f in zip(block, factors, strict=True)
    )
    grid = _block_background(volume, level_block, percentile)
    (z0i, z1i, tz), (y0i, y1i, ty), (x0i, x1i, tx) = (
        _interp_indices(n, g, b)
        for n, g, b in zip(volume.shape, grid.shape, level_block, strict=True)
    )
    ty, tx = ty[:, None], tx[None, :]

    def residual(z0: int, z1: int) -> np.ndarray:
        w = tz[z0:z1, None, None]
        planes = grid[z0i[z0:z1]] * (1 - w) + grid[z1i[z0:z1]] * w
        rows = planes[:, :, x0i] * (1 - tx) + planes[:, :, x1i] * tx
        bg = rows[:, y0i, :] * (1 - ty) + rows[:, y1i, :] * ty
        return volume[z0:z1].astype(np.float32) - bg

    # Noise from every 8th plane: robust standard deviation of the residual.
    sample = np.concatenate(
        [residual(z, z + 1)[0, ::2, ::2].ravel() for z in range(0, volume.shape[0], 8)]
    )
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        sample = np.zeros(1, dtype=np.float32)
    centre = float(np.median(sample))
    noise = 1.4826 * float(np.median(np.abs(sample - centre)))
    floor = centre + noise_floor * noise

    out = np.empty(volume.shape, dtype=np.uint16)

    nz = volume.shape[0]

    def write(z0: int) -> None:
        z1 = min(z0 + slab, nz)
        # One plane of context on each side, for the despeckle neighbours.
        lo, hi = max(z0 - 1, 0), min(z1 + 1, nz)
        r = residual(lo, hi)
        r -= floor
        # NaN (padding) shows as background, inf (float16 overflow on bright
        # beads) as saturated.
        np.nan_to_num(r, copy=False, nan=0.0, posinf=65535.0, neginf=0.0)
        np.clip(r, 0, 65535, out=r)
        if despeckle_neighbours > 0:
            above = r >= 1  # what stays non-zero after the uint16 cast
            if lo == z0:  # outside the volume counts as not set
                above = np.concatenate([np.zeros_like(above[:1]), above])
            if hi == z1:
                above = np.concatenate([above, np.zeros_like(above[:1])])
            inner = r[z0 - lo : z0 - lo + (z1 - z0)]
            inner[_isolated(above, despeckle_neighbours)] = 0
        out[z0:z1] = r[z0 - lo : z0 - lo + (z1 - z0)]

    with ThreadPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        list(pool.map(write, range(0, volume.shape[0], slab)))
    upper = float(np.percentile(out[::4, ::4, ::4], UPPER_PERCENTILE))
    return out, (0.0, max(upper, 1.0))
