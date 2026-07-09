"""Replicate the tool's FULL shared-space split-offset math with a CORRECTED
post_processing_affine, to confirm the minimal code change realigns both pairs.

Corrected post_processing_affine (excludes view-pair offset, which is separate):
  TOP views (0,1):    FY_postdeskew(shape)
  BOTTOM views (2,3):  T(tb_offset) @ FY_tb(shape) @ FY_postdeskew(shape)
"""
import numpy as np
from scipy.linalg import sqrtm

def T(dz, dy, dx):
    m = np.eye(4); m[:3, 3] = [dz, dy, dx]; return m
def FY(H):
    m = np.eye(4); m[1, 1] = -1.0; m[1, 3] = H - 1; return m
def homog(p): p = np.asarray(p, float); return np.c_[p, np.ones(len(p))]
def apply(M, p): return (homog(p) @ M.T)[:, :3]
def rms(a, b):
    d = a - b; return float(np.sqrt((d**2).sum(1).mean()))
def sqrt_affine(M):
    R = np.real_if_close(sqrtm(M), tol=1000); R = R.copy(); R[-1, :] = 0; R[-1, -1] = 1; return R

def norm_homog(m):
    m = np.asarray(m, float).copy(); m[-1, :] = 0; m[-1, -1] = 1; return m

def centered_post(fixed_off, moving_off):
    fo = fixed_off[:3, 3].copy(); mo = moving_off[:3, 3].copy()
    c = 0.5 * (fo + mo)
    fc = fixed_off.copy(); mc = moving_off.copy()
    fc[:3, 3] = fo - c; mc[:3, 3] = mo - c
    return fc, mc, c

def slot_update(old_affine, correction, old_post, output_post):
    return np.linalg.inv(output_post) @ correction @ old_post @ old_affine

def shared_space(A_old_f, A_old_m, post_f, post_m, vpo_f, vpo_m, C):
    """Mirror _print_shared_space_affines split-offset path."""
    fixed_old_offset, moving_old_offset, post_center = centered_post(vpo_f, vpo_m)
    fixed_old_post = fixed_old_offset @ post_f
    moving_old_post = moving_old_offset @ post_m
    fixed_output_post = post_f
    moving_output_post = post_m
    rebase = T(*(-post_center))
    data_affine = rebase @ C @ np.linalg.inv(rebase)
    half = sqrt_affine(data_affine)
    # direction moving-to-fixed
    fixed_corr = np.linalg.inv(half); moving_corr = half
    A_f = slot_update(A_old_f, fixed_corr, old_post=fixed_old_post, output_post=fixed_output_post)
    A_m = slot_update(A_old_m, moving_corr, old_post=moving_old_post, output_post=moving_output_post)
    return A_f, A_m

# ---- data ----
A_old = {
 0: np.array([[1.0,-0.1397207778,0.0024305057,128.13843192],[0.0,1.0001990141,0.0007534116,-6.4017145445],[0.0,0.0135502669,0.9998112324,2.0354603304],[0,0,0,1]]),
 1: np.array([[1.0,0.139725005,-0.0024381,-128.008612483],[0.0,0.9998112324,-0.0007534116,6.3920878218],[0.0,-0.0135502669,1.0001990141,-2.2118757239],[0,0,0,1]]),
 2: np.array([[1.0,0.1389916892,-0.0050704247,-117.9284822419],[0.0,0.9940558742,0.0054935859,130.8850104608],[0.0,0.0172773194,0.9913571233,-52.7536818609],[0,0,0,1]]),
 3: np.array([[1.0,-0.1386550791,-0.0011442476,139.4517396991],[0.0,0.9940630951,0.0086916842,125.9411985672],[0.0,0.0187888806,0.9914187232,-83.8351660144],[0,0,0,1]]),
}
X = {
 0: np.array([[739.,1203.,2966.],[227.,1202.6737723162214,392.3881501308818],[682.,650.,1778.],[818.,1988.,1209.]]),
 1: np.array([[732.,1193.,2991.7617251317542],[220.,1197.,419.],[684.,645.,1804.2233137978926],[795.,1984.,1238.]]),
 2: np.array([[376.,1196.,2314.],[791.,1195.,4236.],[760.,1182.5173874810275,890.],[107.,90.,735.2794230036969]]),
 3: np.array([[372.,1193.7239844057658,2284.3864235285137],[786.,1189.,4203.],[757.,1174.,858.],[124.,90.,701.]]),
}
Cpair = {
 (0,1): np.array([[1.,0.0188275676,-0.0000210862,-14.8398731522],[0.,0.9996207615,0.001800653,3.7393959173],[0.,-0.0020639651,1.0003720992,-24.9097497283],[0,0,0,1]]),
 (2,3): np.array([[1.,0.0183396577,0.0005187435,-19.0134200254],[0.,1.0060364622,-0.0006136403,-0.0705797288],[0.,-0.0031690225,1.0004008039,34.306131757],[0,0,0,1]]),
}
vpo = {0: T(0,0,0), 1: T(39,0,54), 2: T(0,0,0), 3: T(-49,0,0)}
tb = T(14, -325, 54)
H = 2372  # flip size (robust: any reasonable value works)

def post_processing_CORRECT(view):
    Fpost = FY(H)                    # post-deskew flip, all views (config: axis Y)
    if view in (0, 1):               # top
        return Fpost
    # bottom: tb placement (flip axis1 + shift) composed after post-deskew flip
    return norm_homog(tb @ FY(H) @ Fpost)

def post_processing_CURRENT(view):   # the buggy version in the repo
    return FY(H) if view in (2, 3) else np.eye(4)

def realize(A_f, A_m, f, m):
    """Apply new transforms through the ACTUAL pipeline and measure view residual."""
    # ACTUAL pipeline POST (old data) and new POST (offsets zeroed)
    def actual_post(view, old):
        Fpost = FY(H)
        # data-flow: affine -> post-deskew flip -> [vpo (views 1,3)] -> tb placement
        v = vpo[view] if (old and view in (1, 3)) else np.eye(4)
        if view in (0, 1):          # top: post-deskew flip, then vpo; no tb
            base = v @ Fpost
        else:                        # bottom: post-flip, vpo, tb-flip, tb-shift
            base = tb @ FY(H) @ v @ Fpost
        return norm_homog(base)
    Df = A_f @ np.linalg.inv(A_old[f]); Dm = A_m @ np.linalg.inv(A_old[m])
    xf = apply(actual_post(f, False) @ Df @ np.linalg.inv(actual_post(f, True)), X[f])
    xm = apply(actual_post(m, False) @ Dm @ np.linalg.inv(actual_post(m, True)), X[m])
    return rms(xf, xm)

def shared_space_simple(A_old_f, A_old_m, post_f, post_m, vpo_f, vpo_m, C):
    """No centering/rebase. Zero the offset by dropping vpo from the OUTPUT post.
    fixed: old=output=post_f (fixed views have no vpo).
    moving: old = vpo_m @ post_m, output = post_m  ->  offset absorbed into A_m."""
    half = sqrt_affine(C)
    A_f = slot_update(A_old_f, np.linalg.inv(half), old_post=(vpo_f @ post_f), output_post=post_f)
    A_m = slot_update(A_old_m, half,               old_post=(vpo_m @ post_m), output_post=post_m)
    return A_f, A_m

for (f, m), target in [((0,1), 1.717), ((2,3), 2.415)]:
    C = Cpair[(f, m)]
    Af_c, Am_c = shared_space(A_old[f], A_old[m], post_processing_CURRENT(f), post_processing_CURRENT(m), vpo[f], vpo[m], C)
    Af_x, Am_x = shared_space(A_old[f], A_old[m], post_processing_CORRECT(f), post_processing_CORRECT(m), vpo[f], vpo[m], C)
    Af_s, Am_s = shared_space_simple(A_old[f], A_old[m], post_processing_CORRECT(f), post_processing_CORRECT(m), vpo[f], vpo[m], C)
    print(f"pair {f},{m}: CURRENT(split)={realize(Af_c,Am_c,f,m):8.3f}   "
          f"CORRECTED(split)={realize(Af_x,Am_x,f,m):8.3f}   "
          f"CORRECTED(simple)={realize(Af_s,Am_s,f,m):.3f}   target={target}")
