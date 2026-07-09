"""Pair 0,1 (TOP views) check — these get a net post-deskew Y-flip, so this
tests sensitivity to the flip Y-size (view_shape vs canvas)."""
import numpy as np
from scipy.linalg import sqrtm

np.set_printoptions(suppress=True, precision=4)

def T(dz, dy, dx):
    m = np.eye(4); m[:3, 3] = [dz, dy, dx]; return m

def FY(H):
    m = np.eye(4); m[1, 1] = -1.0; m[1, 3] = H - 1; return m

def homog(p): p = np.asarray(p, float); return np.c_[p, np.ones(len(p))]
def apply(M, p): return (homog(p) @ M.T)[:, :3]
def rms(a, b):
    d = a - b; return float(np.sqrt((d**2).sum(1).mean())), np.linalg.norm(d, 1)

def sqrt_affine(M):
    R = np.real_if_close(sqrtm(M), tol=1000); R[-1, :] = 0; R[-1, -1] = 1; return R

A_old = {
 0: np.array([[1.0,-0.1397207778,0.0024305057,128.13843192],
              [0.0,1.0001990141,0.0007534116,-6.4017145445],
              [0.0,0.0135502669,0.9998112324,2.0354603304],[0,0,0,1]]),
 1: np.array([[1.0,0.139725005,-0.0024381,-128.008612483],
              [0.0,0.9998112324,-0.0007534116,6.3920878218],
              [0.0,-0.0135502669,1.0001990141,-2.2118757239],[0,0,0,1]]),
}
A_new = {  # tool log output for pair 0,1
 0: np.array([[1.0,-0.1491387607,0.0024466603,116.1455675201],
              [0.0,1.0003751310,-0.0001466059,-8.2917982535],
              [0.0,0.0145799188,0.9996246543,-12.5230506441],[0,0,0,1]]),
 1: np.array([[1.0,0.1491380948,-0.0024599759,-115.8780556450],
              [0.0,0.9996098958,0.0001472378,8.3129860263],
              [0.0,-0.0145845831,1.0003863255,12.3384401308],[0,0,0,1]]),
}
x0 = np.array([[739.0,1203.0,2966.0],[227.0,1202.6737723162214,392.3881501308818],
               [682.0,650.0,1778.0],[818.0,1988.0,1209.0]])
x1 = np.array([[732.0,1193.0,2991.7617251317542],[220.0,1197.0,419.0],
               [684.0,645.0,1804.2233137978926],[795.0,1984.0,1238.0]])

# estimated correction (moving view1 -> fixed view0), world space, from log
C = np.array([[1.0,0.0188275676,-0.0000210862,-14.8398731522],
              [0.0,0.9996207615,0.001800653,3.7393959173],
              [0.0,-0.0020639651,1.0003720992,-24.9097497283],[0,0,0,1]])
half = sqrt_affine(C)
vpo1 = T(39, 0, 54)   # pair_0_1 offset applied to moving view 1

print("baseline residual x0 vs x1:", round(rms(x0, x1)[0], 3))

# tool-current realized under ACTUAL pipeline post (top views: net flip F, size=H_view)
for H in (2372, 2372 - 325):
    F = FY(H)
    # ACTUAL post: top views get post-deskew flip F; moving also view-pair offset (old)
    POST0 = F
    POST1_OLD = vpo1 @ F
    POST1_NEW = F
    D0 = A_new[0] @ np.linalg.inv(A_old[0])
    D1 = A_new[1] @ np.linalg.inv(A_old[1])
    x0n = apply(POST0 @ D0 @ np.linalg.inv(POST0), x0)
    x1n = apply(POST1_NEW @ D1 @ np.linalg.inv(POST1_OLD), x1)
    print(f"[H={H}] TOOL-CURRENT A_new realized under actual pipeline: RMS={rms(x0n,x1n)[0]:.3f}")

    # FIXED formula: compute A_new with the SAME (actual) post
    A0f = np.linalg.inv(POST0)     @ np.linalg.inv(half) @ POST0     @ A_old[0]
    A1f = np.linalg.inv(POST1_NEW) @ half                @ POST1_OLD @ A_old[1]
    D0f = A0f @ np.linalg.inv(A_old[0]); D1f = A1f @ np.linalg.inv(A_old[1])
    x0f = apply(POST0 @ D0f @ np.linalg.inv(POST0), x0)
    x1f = apply(POST1_NEW @ D1f @ np.linalg.inv(POST1_OLD), x1)
    print(f"[H={H}] FIXED formula realized under actual pipeline:      RMS={rms(x0f,x1f)[0]:.3f}")
print("point-picker reported RMS for pair 0,1 = 1.717")
