"""Block-wise phase cross-correlation between two deskewed views of one camera.

Robust for dense bead fields and shear: split the overlap crop into sub-blocks,
measure the local (dz,dy,dx) shift that best aligns view A to view B in each
block, and report the distribution of shift magnitudes. Aligned views -> ~0-2px.

Usage: python measure_xcorr.py <deskewed.zarr/0> <chA> <chB>
"""
import sys
import numpy as np
import zarr
from skimage.registration import phase_cross_correlation

zpath, ca, cb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Z0, Z1, Y0, Y1, X0, X1 = 450, 850, 850, 1650, 1900, 3300
arr = zarr.open(zpath, mode="r")
va = np.asarray(arr[0, ca, Z0:Z1, Y0:Y1, X0:X1], dtype=np.float32)
vb = np.asarray(arr[0, cb, Z0:Z1, Y0:Y1, X0:X1], dtype=np.float32)

# sub-blocks
bz, by, bx = 200, 200, 350
mags, shifts = [], []
for z in range(0, va.shape[0] - bz + 1, bz):
    for y in range(0, va.shape[1] - by + 1, by):
        for x in range(0, va.shape[2] - bx + 1, bx):
            a = va[z:z+bz, y:y+by, x:x+bx]
            b = vb[z:z+bz, y:y+by, x:x+bx]
            if a.max() < 100 or b.max() < 100:
                continue  # skip empty blocks
            sh, err, _ = phase_cross_correlation(a, b, upsample_factor=10)
            mag = float(np.linalg.norm(sh))
            if mag < 60:  # ignore blocks where corr failed (ambiguous)
                mags.append(mag); shifts.append(sh)

mags = np.array(mags)
print(f"blocks with signal & valid corr: {len(mags)}")
if len(mags):
    print(f"MEAN   |shift| = {mags.mean():.3f} px")
    print(f"MEDIAN |shift| = {np.median(mags):.3f} px")
    print(f"per-axis median (dz,dy,dx) = {np.round(np.median(np.array(shifts), axis=0), 2)}")
    print(f"shift magnitudes: {np.round(np.sort(mags), 1)}")
