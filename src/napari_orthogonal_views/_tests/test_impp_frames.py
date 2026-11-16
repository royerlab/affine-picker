"""The impp frame model: a bead fit recovers a known view matrix or camera alignment, in the calibration file's frame."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))
import impp_frames as frames  # noqa: E402

SHAPE = (60, 400, 300)
RAW = (80, 500, 300)


def _deskew(rng):
    m = np.eye(4)
    m[0, 1] = 0.26 + rng.normal(0, 0.01)
    m[1:3, 1:3] += rng.normal(0, 0.01, (2, 2))
    m[:3, 3] = rng.normal(0, 5, 3)
    return m


def _frame(rng, view, *, ratio=1.0, top_shift=0.0, mirrored=False, origin=(0, 0, 0), camera=None, file=None):
    file = _deskew(rng) if file is None else file
    f = np.diag([1 / ratio, 1, 1, 1])
    sh = frames.translation((0, -top_shift, 0)) if view in (2, 3) else np.eye(4)
    camera = frames.embed_camera([[1.001, 0.002, 3.0], [-0.001, 0.999, -2.0], [0, 0, 1]]) if camera is None else camera
    return frames.ViewFrame(pre=frames.flip(RAW, (2,)), camera=camera, deskew_loaded=sh @ f @ file @ np.linalg.inv(f),
                            deskew_file=file, after=frames.flip(SHAPE, (1,)), frame_ratio=ratio, top_shift=top_shift,
                            view=view, shape=SHAPE, mirrored=mirrored, origin=np.asarray(origin, float))


def _world(frame, raw):
    return frames.apply(frames.translation(frame.origin) @ frame.mirror() @ frame.raw_to_view(), raw)


@pytest.mark.parametrize("view,ratio,shift,mirrored", [(1, 1.0, 0.0, False), (3, 8.0, -232.0, True)])
def test_a_slot_solve_recovers_the_file_matrix_that_puts_b_on_a(view, ratio, shift, mirrored):
    rng = np.random.default_rng(view)
    old = _frame(rng, view, ratio=ratio, top_shift=shift, mirrored=mirrored, origin=(1.5, -20, 4))
    true_file = old.deskew_file @ frames.translation((0.3, -1.2, 0.8))
    true_file[1, 2] += 0.004
    true = _frame(rng, view, ratio=ratio, top_shift=shift, mirrored=mirrored, origin=old.origin, camera=old.camera,
                  file=true_file)
    raw = rng.uniform((0, 0, 0), RAW, (30, 3))
    world_b, world_a = _world(old, raw), _world(true, raw)    # where B shows a bead, and where A shows it
    w = frames.apply(np.eye(4), world_b)
    design = np.c_[w, np.ones(len(w))]
    solution, *_ = np.linalg.lstsq(design, world_a, rcond=None)
    world = np.eye(4)
    world[:3, :] = solution.T
    assert np.allclose(frames.solve_slot(old, world), true_file, atol=1e-6)


def test_a_camera_fit_recovers_the_alignment_in_raw_and_ignores_nothing_else():
    rng = np.random.default_rng(5)
    old = _frame(rng, 0, mirrored=False, origin=(0, 0, 0))
    true_camera = np.array([[1.004, -0.003, 11.0], [0.002, 0.997, -4.0], [0, 0, 1]])
    true = _frame(rng, 0, camera=frames.embed_camera(true_camera), file=old.deskew_file)
    raw = rng.uniform((0, 0, 0), RAW, (25, 3))
    fit = frames.camera_fit(old, _world(true, raw), _world(old, raw))
    assert np.allclose(fit.matrix, true_camera, atol=1e-6)
    assert fit.frame_rms < 1e-6 and max(fit.rms()) < 1e-6


def test_a_frame_that_does_not_reproduce_impps_geometry_is_refused():
    rng = np.random.default_rng(2)
    frame = _frame(rng, 2, ratio=2.0, top_shift=-100.0)
    frame.check(frame.raw_to_view())
    with pytest.raises(SystemExit, match="raw_to_view"):
        frame.check(frame.raw_to_view() @ frames.translation((0, 1, 0)))
    wrong = frames.ViewFrame(**{**frame.__dict__, "top_shift": 0.0})
    with pytest.raises(SystemExit, match="load"):
        wrong.check(wrong.raw_to_view())


def test_the_camera_key_is_found_by_the_matrix_impp_applied():
    applied = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0, 0, 1]])
    transforms = {"camera_0": {"view_0": np.eye(3).tolist()}, "camera_1": {"view_0": applied.tolist()}}
    assert frames.camera_key(transforms, 0, 0, applied) == "camera_1"
    with pytest.raises(SystemExit):
        frames.camera_key(transforms, 0, 0, applied * 2)
