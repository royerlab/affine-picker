"""End-to-end test of the PATCHED load_zarr_views_for_registration code:
build real pipeline slots from the real project config, run the shared-space
solve on the saved point-picker correction, then realize the emitted
view_transforms through the actual pipeline and confirm the residual matches
the point picker."""
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

sys.argv = ["x"]  # keep argparse quiet on import
import load_zarr_views_for_registration as L  # noqa: E402

PROJ = Path("/hpc/projects/group.royer/imaging/dev/2025_09_12_daxi_Daxi_B")
SHAPE = (1337, 2372, 4501)
H = SHAPE[1]


def T(dz, dy, dx):
    m = np.eye(4); m[:3, 3] = [dz, dy, dx]; return m
def FY(h):
    m = np.eye(4); m[1, 1] = -1.0; m[1, 3] = h - 1; return m
def homog(p): p = np.asarray(p, float); return np.c_[p, np.ones(len(p))]
def apply(M, p): return (homog(p) @ M.T)[:, :3]
def rms(a, b): d = a - b; return float(np.sqrt((d**2).sum(1).mean()))

tb = T(14, -325, 54); vpo = {0: T(0,0,0), 1: T(39,0,54), 2: T(0,0,0), 3: T(-49,0,0)}
expected_post = {0: FY(H), 1: vpo[1] @ FY(H), 2: tb, 3: vpo[3] @ tb}

print("--- check _infer_pipeline_slot post_affine matches the real pipeline ---")
slots = {}
for v in (0, 1, 2, 3):
    slot = L._infer_pipeline_slot(
        alignment_mode="view", project_folder=PROJ, transform_yaml=None,
        view_id=v, camera_id=3, view_shape_zyx=SHAPE,
        allow_identity_old_transform=False,
    )
    slots[v] = slot
    ok = np.allclose(slot.post_affine, expected_post[v], atol=1e-6)
    print(f"view {v}: post_affine matches pipeline = {ok}")
    if not ok:
        print(" got\n", np.round(slot.post_affine, 3), "\n exp\n", np.round(expected_post[v], 3))

A_old = {v: slots[v].old_affine for v in (0, 1, 2, 3)}
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


class FakeManager:
    def __init__(self, C): self._C = C
    def get_estimated_affine(self): return self._C


def parse_view_transforms(text):
    """Pull the emitted 4x4 matrices out of the copy-paste YAML block."""
    out = {}
    m = re.search(r"^view_transforms:\n(.*?)(?:\n\[|\nview_pair_offsets:|\Z)", text, re.S | re.M)
    block = m.group(1)
    for vm in re.finditer(r"^  (\d+):\n((?:    - \[.*\]\n?)+)", block, re.M):
        vid = int(vm.group(1))
        rows = re.findall(r"\[([^\]]+)\]", vm.group(2))
        out[vid] = np.array([[float(x) for x in r.split(",")] for r in rows])
    return out


def actual_post(view, old):
    v = vpo[view] if (old and view in (1, 3)) else np.eye(4)
    base = (v @ FY(H)) if view in (0, 1) else (tb @ FY(H) @ v @ FY(H))
    base = np.asarray(base, float).copy(); base[-1, :] = 0; base[-1, -1] = 1
    return base


print("\n--- full shared-space solve via patched code, realized on beads ---")
for (f, m), target in [((0, 1), 1.717), ((2, 3), 2.415)]:
    inf = L.TransformInference(
        fixed_camera_id=3, moving_camera_id=3, fixed_view_id=f, moving_view_id=m,
        fixed_slot=slots[f], moving_slot=slots[m], output_mode="shared-space",
        invert_estimated_affine=False, output_inverse=False,
        show_full_pipeline_matrices=False, shared_space_offset_mode="split-offset",
        fixed_display_affine=np.eye(4), moving_display_affine=np.eye(4),
        fixed_view_source="t", moving_view_source="t",
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        L._print_yaml_ready_affine(FakeManager(Cpair[(f, m)]), inf)
    emitted = parse_view_transforms(buf.getvalue())
    Df = emitted[f] @ np.linalg.inv(A_old[f]); Dm = emitted[m] @ np.linalg.inv(A_old[m])
    xf = apply(actual_post(f, False) @ Df @ np.linalg.inv(actual_post(f, True)), X[f])
    xm = apply(actual_post(m, False) @ Dm @ np.linalg.inv(actual_post(m, True)), X[m])
    print(f"pair {f},{m}: realized RMS = {rms(xf, xm):.3f}   target(point-picker) = {target}")

print("\n--- SHENG SCENARIO: realize with new config tb=0, vpo=0 (daxi-b config lacks them) ---")
def new_post_shengcfg(view):
    # new run: no top_bottom_offsets, no view_pair_offsets in daxi-b config
    base = FY(H) if view in (0, 1) else (FY(H) @ FY(H))  # bottom: tb=0 -> flips cancel -> I
    base = np.asarray(base, float).copy(); base[-1, :] = 0; base[-1, -1] = 1
    return base
for (f, m), target in [((0, 1), 1.717), ((2, 3), 2.415)]:
    inf = L.TransformInference(
        fixed_camera_id=3, moving_camera_id=3, fixed_view_id=f, moving_view_id=m,
        fixed_slot=slots[f], moving_slot=slots[m], output_mode="shared-space",
        invert_estimated_affine=False, output_inverse=False,
        show_full_pipeline_matrices=False, shared_space_offset_mode="split-offset",
        fixed_display_affine=np.eye(4), moving_display_affine=np.eye(4),
        fixed_view_source="t", moving_view_source="t")
    buf = io.StringIO()
    with redirect_stdout(buf):
        L._print_yaml_ready_affine(FakeManager(Cpair[(f, m)]), inf)
    emitted = parse_view_transforms(buf.getvalue())
    Df = emitted[f] @ np.linalg.inv(A_old[f]); Dm = emitted[m] @ np.linalg.inv(A_old[m])
    # old post = 2025_09_12 project (tb + vpo baked in); new post = sheng config (tb=0,vpo=0)
    xf = apply(new_post_shengcfg(f) @ Df @ np.linalg.inv(actual_post(f, True)), X[f])
    xm = apply(new_post_shengcfg(m) @ Dm @ np.linalg.inv(actual_post(m, True)), X[m])
    print(f"pair {f},{m}: realized RMS = {rms(xf, xm):.3f}   target = {target}")
