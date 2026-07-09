"""Offline check: does the tool's output A_new actually align views under the
REAL pipeline POST chain?

Pipeline per view (deskew-only), data-flow order applied to raw:
  pre-flip -> resample_affine(A @ camera_align) -> post-deskew Y-flip (all views)
  -> view-pair offset shift (views 1,3 only) -> top/bottom placement
     (bottom views 2,3: extra Y-flip [cancels the post-deskew flip] + tb_offset shift)

For a matched bead, displayed coord x_v = W_v @ p, with
  W_v = POST_v @ A_v @ PRE_v   (PRE = camera_align @ pre_flip, cancels in updates)

New coord after updating A_v_old -> A_v_new and zeroing view-pair offset:
  x_v_new = POST_NEW_v @ A_v_new @ inv(A_v_old) @ inv(POST_OLD_v) @ x_v
"""
import numpy as np

np.set_printoptions(suppress=True, precision=4)


def T(dz, dy, dx):
    m = np.eye(4)
    m[:3, 3] = [dz, dy, dx]
    return m


def homog(pts):
    pts = np.asarray(pts, float)
    return np.c_[pts, np.ones(len(pts))]


def apply(M, pts):
    return (homog(pts) @ M.T)[:, :3]


# --- A_old (impp_settings) and A_new (tool log output), views 2 & 3 (z,y,x) ---
A_old = {
    2: np.array([[1.0, 0.1389916892, -0.0050704247, -117.9284822419],
                 [0.0, 0.9940558742, 0.0054935859, 130.8850104608],
                 [0.0, 0.0172773194, 0.9913571233, -52.7536818609],
                 [0, 0, 0, 1]]),
    3: np.array([[1.0, -0.1386550791, -0.0011442476, 139.4517396991],
                 [0.0, 0.9940630951, 0.0086916842, 125.9411985672],
                 [0.0, 0.0187888806, 0.9914187232, -83.8351660144],
                 [0, 0, 0, 1]]),
}
A_new = {
    2: np.array([[1.0, 0.1480620177, -0.0052814875, -104.3519000772],
                 [0.0, 0.9910645411, 0.0051743761, 137.6029508755],
                 [0.0, 0.0157063422, 0.9911505691, -66.3585589284],
                 [0, 0, 0, 1]]),
    3: np.array([[1.0, -0.1477520594, -0.0009653202, 125.9773783359],
                 [0.0, 0.9970644085, 0.0090215754, 119.1827624913],
                 [0.0, 0.0203652184, 0.9916308953, -70.2522352431],
                 [0, 0, 0, 1]]),
}

x2 = [[376.0, 1196.0, 2314.0], [791.0, 1195.0, 4236.0],
      [760.0, 1182.5173874810275, 890.0], [107.0, 90.0, 735.2794230036969]]
x3 = [[372.0, 1193.7239844057658, 2284.3864235285137], [786.0, 1189.0, 4203.0],
      [757.0, 1174.0, 858.0], [124.0, 90.0, 701.0]]
x2 = np.array(x2); x3 = np.array(x3)

# project views_alignment.yaml
tb = T(14, -325, 54)           # top/bottom offset, applied to bottom views (2,3)
vpo3 = T(-49, 0, 0)            # pair_2_3 view-pair offset, applied to moving view 3

# Pair 2,3 are BOTH bottom -> post-deskew Y-flip cancels with tb Y-flip -> no net flip.
# POST_2 = tb  (fixed, no view-pair offset)
# POST_3_OLD = tb @ vpo3 ; POST_3_NEW = tb (offset zeroed by tool)
POST2 = tb
POST3_OLD = tb @ vpo3
POST3_NEW = tb

def rms(a, b):
    d = a - b
    return float(np.sqrt((d ** 2).sum(axis=1).mean())), np.linalg.norm(d, axis=1)

print("=== BASELINE: old displayed residual x2 vs x3 (before any update) ===")
r, n = rms(x2, x3); print(f"RMS={r:.3f}  per-pair={np.round(n,3)}")

# realized new positions under ACTUAL post
D2 = A_new[2] @ np.linalg.inv(A_old[2])
D3 = A_new[3] @ np.linalg.inv(A_old[3])
x2n = apply(POST2 @ D2 @ np.linalg.inv(POST2), x2)
x3n = apply(POST3_NEW @ D3 @ np.linalg.inv(POST3_OLD), x3)
print("\n=== REALIZED residual after tool A_new, ACTUAL pipeline POST ===")
r, n = rms(x2n, x3n); print(f"RMS={r:.3f}  per-pair={np.round(n,3)}")
print("(point-picker reported RMS was 2.415 — this is the target)")

# For contrast: realized residual under the TOOL'S ASSUMED post
# tool POST_2 = objective Y-flip on canvas Y=2372 ; POST_3 = vpo(centered) @ Yflip
# split-offset: fixed centered +24.5 z, moving -24.5 z, output 0
def FY(H):
    m = np.eye(4); m[1, 1] = -1.0; m[1, 3] = H - 1
    return m
H = 2372
tool_POST2_old = T(24.5, 0, 0) @ FY(H)   # centered fixed offset + objective flip
tool_POST3_old = T(-24.5, 0, 0) @ FY(H)
tool_POST2_new = FY(H)                    # output offset zero
tool_POST3_new = FY(H)
x2t = apply(tool_POST2_new @ D2 @ np.linalg.inv(tool_POST2_old), x2)
x3t = apply(tool_POST3_new @ D3 @ np.linalg.inv(tool_POST3_old), x3)
print("\n=== SANITY: realized residual under the TOOL'S OWN assumed POST ===")
r, n = rms(x2t, x3t); print(f"RMS={r:.3f}  per-pair={np.round(n,3)}")
print("(should be ~2.4 by construction if the tool is self-consistent)")

# ============================================================
# FIX: recompute A_new using the ACTUAL pipeline POST, verify realized ~2.4
# ============================================================
from scipy.linalg import sqrtm

def sqrt_affine(M):
    R = np.real_if_close(sqrtm(M), tol=1000)
    R[-1, :] = 0.0; R[-1, -1] = 1.0
    return R

# estimated correction (moving view3 -> fixed view2), world space, from log
C = np.array([[1.0, 0.0183396577, 0.0005187435, -19.0134200254],
              [0.0, 1.0060364622, -0.0006136403, -0.0705797288],
              [0.0, -0.0031690225, 1.0004008039, 34.306131757],
              [0, 0, 0, 1]])
half = sqrt_affine(C)

# correct POST (actual pipeline). pair 2,3 both bottom: net no flip.
POST2_OLD = tb;              POST2_NEW = tb           # fixed, no view-pair offset
POST3_OLD = tb @ vpo3;       POST3_NEW = tb           # moving, offset zeroed in output

# shared-space half split: fixed<-inv(half), moving<-half
A2_fix = np.linalg.inv(POST2_NEW) @ np.linalg.inv(half) @ POST2_OLD @ A_old[2]
A3_fix = np.linalg.inv(POST3_NEW) @ half              @ POST3_OLD @ A_old[3]

# realize under actual pipeline POST (independent)
D2f = A2_fix @ np.linalg.inv(A_old[2])
D3f = A3_fix @ np.linalg.inv(A_old[3])
x2f = apply(POST2_NEW @ D2f @ np.linalg.inv(POST2_OLD), x2)
x3f = apply(POST3_NEW @ D3f @ np.linalg.inv(POST3_OLD), x3)
print("\n=== FIXED formula (correct POST): realized residual under actual pipeline ===")
r, n = rms(x2f, x3f); print(f"RMS={r:.3f}  per-pair={np.round(n,3)}")
print("target = point-picker RMS 2.415")
