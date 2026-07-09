"""Measure the mean bead mismatch between two deskewed views of the same camera.

Robust to large displacements + dense fields: detect bright beads, keep only
*isolated* ones (no same-view neighbor within ISO px), then match across views
within a generous radius and require the match to be mutually unambiguous.

Usage: python measure_diff.py <deskewed.zarr/0> <fixed_channel> <moving_channel>
"""
import sys
import numpy as np
import zarr
from scipy.ndimage import maximum_filter, label, center_of_mass
from scipy.spatial import cKDTree

zpath, cf, cm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Z0, Z1, Y0, Y1, X0, X1 = 450, 850, 850, 1650, 1900, 3300
ISO = 45.0      # min same-view separation to call a bead "isolated"
RADIUS = 120.0  # cross-view search radius (must exceed the true displacement)

arr = zarr.open(zpath, mode="r")

def load(ch):
    return np.asarray(arr[0, ch, Z0:Z1, Y0:Y1, X0:X1], dtype=np.float32)

def detect(vol, min_sep=5):
    thr = max(150.0, np.percentile(vol, 99.97))
    mx = maximum_filter(vol, size=min_sep)
    lbl, n = label((vol == mx) & (vol >= thr))
    if n == 0:
        return np.zeros((0, 3))
    return np.array(center_of_mass(vol, lbl, range(1, n + 1)))

def isolated(pts):
    if len(pts) < 2:
        return pts
    t = cKDTree(pts)
    d, _ = t.query(pts, k=2)
    return pts[d[:, 1] > ISO]

vf, vm = load(cf), load(cm)
pf, pm = detect(vf), detect(vm)
pf_i, pm_i = isolated(pf), isolated(pm)
print(f"detected fixed={len(pf)} (isolated {len(pf_i)}), moving={len(pm)} (isolated {len(pm_i)})")

if len(pf_i) and len(pm_i):
    tree = cKDTree(pm_i)
    d, idx = tree.query(pf_i, k=1)
    keep = d < RADIUS
    m = d[keep]
    print(f"matched isolated pairs within {RADIUS:.0f}px: {keep.sum()} / {len(pf_i)}")
    if keep.sum():
        print(f"MEAN   bead difference = {m.mean():.3f} px")
        print(f"MEDIAN bead difference = {np.median(m):.3f} px")
        print(f"RMS    bead difference = {np.sqrt((m**2).mean()):.3f} px")
        vec = pf_i[keep] - pm_i[idx[keep]]
        print(f"median (dz,dy,dx)      = {np.round(np.median(vec, axis=0), 2)}")
