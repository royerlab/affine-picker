import json
from functools import partial

import numpy as np
import pytest

from napari_orthogonal_views import (
    downsampled_layer_kwargs,
    estimate_affine_from_points,
    estimate_affine_from_points_components,
)
from napari_orthogonal_views.fit_diagnostics import apply_affine, evaluate_fit
from napari_orthogonal_views.point_picker_widget import PointPickerWidget

TRUE_AFFINE = np.array(
    [
        [1.0, 0.01, 0.0, 5.0],
        [0.0, 0.99, 0.02, -3.0],
        [0.0, 0.0, 1.01, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _pairs(n=8, seed=0, extent=(40, 60, 80)):
    rng = np.random.default_rng(seed)
    moving = rng.uniform(0, extent, size=(n, 3))
    fixed = apply_affine(TRUE_AFFINE, moving)
    ids = list(range(1, n + 1))
    return dict(zip(ids, fixed, strict=True)), dict(
        zip(ids, moving, strict=True)
    )


# ---------------------------------------------------------- fit diagnostics


def test_exact_pairs_fit_with_zero_residual():
    fixed, moving = _pairs()
    report = evaluate_fit(estimate_affine_from_points, fixed, moving, fixed)
    assert report.ok
    np.testing.assert_allclose(report.affine, TRUE_AFFINE, atol=1e-9)
    assert report.rms < 1e-9
    assert report.loo_rms < 1e-6


def test_leave_one_out_exposes_mismatched_pair():
    fixed, moving = _pairs(n=15)
    fixed[3] = fixed[3] + np.array([0.0, 15.0, 0.0])
    report = evaluate_fit(estimate_affine_from_points, fixed, moving, fixed)
    assert report.worst_loo_pair == 3
    # The full fit absorbs part of the error, so the bad pair's own residual
    # understates it; the leave-one-out residual does not.
    assert report.out_of_sample[3] > report.residual_norms[3]


def test_skipped_pair_gets_out_of_sample_residual():
    fixed, moving = _pairs(n=8)
    fixed[3] = fixed[3] + np.array([0.0, 15.0, 0.0])
    used = [i for i in fixed if i != 3]
    report = evaluate_fit(estimate_affine_from_points, fixed, moving, used)
    assert report.n_used == 7
    assert report.rms < 1e-9
    assert report.out_of_sample[3] == pytest.approx(15.0, rel=0.05)


def test_too_few_pairs_reports_how_many_are_needed():
    fixed, moving = _pairs(n=2)
    report = evaluate_fit(estimate_affine_from_points, fixed, moving, fixed)
    assert not report.ok
    assert report.pairs_needed == 2


def test_estimator_failure_is_reported_not_raised():
    def broken(src, tgt):
        raise ValueError("no")

    fixed, moving = _pairs()
    report = evaluate_fit(broken, fixed, moving, fixed)
    assert not report.ok
    assert "no" in report.error


def test_coplanar_picks_are_flagged():
    fixed, moving = _pairs(n=8)
    for i in moving:
        moving[i][0] = 10.0 + 0.01 * i
        fixed[i] = apply_affine(TRUE_AFFINE, moving[i][None])[0]
    report = evaluate_fit(estimate_affine_from_points, fixed, moving, fixed)
    assert any("coplanar" in w and "along Z" in w for w in report.warnings)


def test_stability_grows_with_pick_noise():
    fixed, moving = _pairs(n=6)
    corners = np.array([[0, 0, 0], [40, 60, 80]], dtype=float)
    small = evaluate_fit(
        estimate_affine_from_points,
        fixed,
        moving,
        fixed,
        volume_corners=corners,
        pick_sigma=0.1,
    )
    large = evaluate_fit(
        estimate_affine_from_points,
        fixed,
        moving,
        fixed,
        volume_corners=corners,
        pick_sigma=2.0,
    )
    assert 0 < small.stability < large.stability


def test_implausible_fit_is_flagged():
    fixed, moving = _pairs(n=8)
    fixed = {i: 1.5 * p for i, p in fixed.items()}
    report = evaluate_fit(estimate_affine_from_points, fixed, moving, fixed)
    assert any("implausible" in w for w in report.warnings)


# ------------------------------------------------------------------ widget


@pytest.fixture
def picker(make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    shape = (40, 60, 80)
    viewer.add_image(np.zeros(shape, np.float32), name="Fixed")
    viewer.add_image(np.zeros(shape, np.float32), name="Moving")
    widget = PointPickerWidget(
        viewer,
        "Fixed",
        "Moving",
        autosave_dir=tmp_path / "autosave",
        backup_dir=tmp_path / "backup",
    )
    qtbot.addWidget(widget)
    yield widget
    for preview in widget._preview_viewers:
        preview.close()


def _pick(widget, fixed_world, moving_world):
    widget.add_pair()
    widget.viewer.dims.point = tuple(fixed_world)
    widget.record_fixed()
    widget.viewer.dims.point = tuple(moving_world)
    widget.record_moving()


def _pick_translation(widget, shift, points):
    for p in points:
        _pick(widget, np.add(p, shift), p)


GRID_POINTS = [
    (2, 3, 4),
    (30, 5, 70),
    (10, 50, 20),
    (35, 40, 60),
    (5, 45, 75),
    (20, 20, 10),
]


def test_hotkey_methods_build_pairs(picker):
    _pick_translation(picker, (2, 3, -1), GRID_POINTS)
    report = picker.refresh_fit()
    assert len(picker.point_pairs) == 6
    assert list(picker.point_pairs) == [1, 2, 3, 4, 5, 6]
    assert report.ok and report.rms < 1e-6
    np.testing.assert_allclose(report.affine[:3, 3], (2, 3, -1), atol=1e-6)
    assert picker.apply_button.isEnabled()


def test_record_without_active_pair_creates_one(picker):
    picker.viewer.dims.point = (1, 2, 3)
    picker.record_fixed()
    assert picker.active_pair_id == 1
    assert picker.point_pairs[1].layer1_coords == (1, 2, 3)


def test_undo_reverts_record_and_delete(picker):
    _pick(picker, (1, 1, 1), (2, 2, 2))
    picker.viewer.dims.point = (5, 5, 5)
    picker.record_fixed()
    picker.undo()
    assert picker.point_pairs[1].layer1_coords == (1, 1, 1)
    picker._delete_pair(1)
    assert not picker.point_pairs
    picker.undo()
    assert picker.point_pairs[1].layer2_coords == (2, 2, 2)


def test_apply_disabled_with_reason_until_enough_pairs(picker):
    _pick_translation(picker, (1, 0, 0), GRID_POINTS[:3])
    picker.refresh_fit()
    assert not picker.apply_button.isEnabled()
    assert "1 more" in picker.apply_button.toolTip()
    assert "Need 1 more complete pair" in picker.fit_label.text()


def test_skipped_pair_is_left_out_of_fit_and_output(picker):
    _pick_translation(picker, (2, 3, -1), GRID_POINTS)
    pair = picker.point_pairs[4]
    pair.layer1_coords = tuple(np.add(pair.layer1_coords, (0, 9, 0)))
    picker.set_active_pair(4)
    picker.toggle_active_use()
    report = picker.refresh_fit()
    assert report.n_used == 5 and report.rms < 1e-6
    assert report.out_of_sample[4] == pytest.approx(9.0)
    assert len(picker.get_point_pairs()["Fixed"]) == 5
    assert len(picker.get_point_pairs(include_skipped=True)["Fixed"]) == 6


def test_table_shows_pair_numbers_and_flags_residuals(picker):
    _pick_translation(picker, (2, 3, -1), GRID_POINTS)
    pair = picker.point_pairs[2]
    pair.layer1_coords = tuple(np.add(pair.layer1_coords, (0, 9, 0)))
    picker._pairs_changed()
    picker.refresh_fit()
    row = picker._row_of(2)
    assert picker.table.item(row, 0).text() == "2"
    assert picker.table.item(row, 5).text() != ""
    assert picker.table.item(row, 0).background().color().red() == 192


def test_order_by_loo_puts_worst_pair_first(picker):
    _pick_translation(picker, (2, 3, -1), GRID_POINTS)
    pair = picker.point_pairs[5]
    pair.layer1_coords = tuple(np.add(pair.layer1_coords, (0, 9, 0)))
    picker._pairs_changed()
    picker.order_combo.setCurrentIndex(2)
    picker.refresh_fit()
    assert picker.table.item(0, 0).text() == "5"


def test_save_and_load_round_trip_keeps_unfinished_and_skipped(
    picker, tmp_path
):
    _pick_translation(picker, (2, 3, -1), GRID_POINTS[:4])
    picker.add_pair()
    picker.viewer.dims.point = (7, 8, 9)
    picker.record_fixed()
    picker.set_pair_enabled(2, False)
    path = picker.save_points(tmp_path / "picks.json")

    payload = json.loads(path.read_text())
    assert payload["format_version"] == 2
    assert payload["estimator"] == "estimate_affine_from_points"
    assert len(payload["pair_records"]) == 5
    assert len(payload["pairs"]["Fixed"]) == 3  # complete and used only

    picker.clear_pairs()
    assert picker.load_points(path) == []
    assert len(picker.point_pairs) == 5
    assert picker.point_pairs[2].enabled is False
    assert picker.point_pairs[5].layer2_coords is None
    assert picker.point_pairs[5].layer1_coords == (7, 8, 9)


def test_load_warns_when_source_differs(picker, tmp_path):
    picker.viewer.layers["Moving"].metadata["channel"] = 6
    _pick_translation(picker, (2, 3, -1), GRID_POINTS[:4])
    path = picker.save_points(tmp_path / "picks.json")
    picker.viewer.layers["Moving"].metadata["channel"] = 7
    warnings = picker.load_points(path)
    assert any("channel" in w and "6" in w and "7" in w for w in warnings)


def test_legacy_file_loads(picker, tmp_path):
    legacy = {
        "layer1_name": "Old fixed",
        "layer2_name": "Old moving",
        "pairs": {
            "Old fixed": [[1, 2, 3], [4, 5, 6]],
            "Old moving": [[1, 2, 4], [4, 5, 7]],
        },
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    picker.load_points(path)
    assert picker.point_pairs[2].layer2_coords == (4, 5, 7)


def test_autosave_written_after_edit(picker):
    _pick(picker, (1, 1, 1), (2, 2, 2))
    assert picker.autosave_path.exists()
    records = json.loads(picker.autosave_path.read_text())["pair_records"]
    assert records[0]["moving"] == [2, 2, 2]


def test_backup_copy_written_after_every_edit(picker):
    _pick(picker, (1, 1, 1), (2, 2, 2))
    backup = picker.backup_path
    assert backup.parent.name == "backup"
    assert json.loads(backup.read_text())["pair_records"][0]["moving"] == [
        2,
        2,
        2,
    ]
    picker._delete_pair(1)
    assert json.loads(backup.read_text())["pair_records"] == []


def test_backup_still_written_when_autosave_fails(picker, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    picker.autosave_path = blocker / "picks.json"
    _pick(picker, (1, 1, 1), (2, 2, 2))
    assert picker.backup_path.exists()
    assert blocker / "picks.json" in picker._save_failures


def test_canvas_overlay_lists_hotkeys_and_active_pair(picker):
    picker.set_canvas_overlay_active(True)
    _pick(picker, (1, 1, 1), (2, 2, 2))
    overlay = picker.viewer.text_overlay
    assert overlay.visible
    assert "pair 1, fixed set, moving set" in overlay.text
    assert "Q set fixed point" in overlay.text
    picker.canvas_keys_checkbox.setChecked(False)
    assert not overlay.visible


# ---------------------------------------------------- full-resolution units


@pytest.fixture
def coarse_picker(make_napari_viewer, qtbot, tmp_path):
    """Both layers show a 4x downsampled level; world = full-res voxels."""
    viewer = make_napari_viewer()
    kwargs = downsampled_layer_kwargs(4)
    moving = np.zeros((20, 20, 20), np.float32)
    moving[5, 6, 7] = 1.0
    viewer.add_image(np.zeros((20, 20, 20), np.float32), name="F", **kwargs)
    viewer.add_image(moving, name="M", **kwargs)
    widget = PointPickerWidget(
        viewer, "F", "M", autosave_dir=None, backup_dir=tmp_path / "backup"
    )
    qtbot.addWidget(widget)
    yield widget
    for preview in widget._preview_viewers:
        preview.close()


def _coarse_centre(index):
    return 4 * np.asarray(index, dtype=float) + 1.5


def test_downsampled_picks_are_full_resolution_voxels(coarse_picker):
    coarse_picker.viewer.dims.point = tuple(_coarse_centre((2, 3, 4)))
    coarse_picker.record_fixed()
    data = coarse_picker.get_point_pairs(include_skipped=True)
    assert data["F"] == []  # moving point not set yet
    fixed = coarse_picker._world_to_original_data(
        coarse_picker.point_pairs[1].layer1_coords, "F"
    )
    # Coarse voxel 2 covers full-res voxels 8..11, centre 9.5.
    assert fixed == (9.5, 13.5, 17.5)
    assert "full-resolution voxels" in coarse_picker.layers_label.text()


def test_apply_on_downsampled_layer_moves_data_by_the_fit(coarse_picker):
    shift = np.array([8.0, -4.0, 4.0])  # whole coarse voxels
    for index in GRID_POINTS:
        p = _coarse_centre(np.asarray(index) % 18)
        _pick(coarse_picker, p + shift, p)
    report = coarse_picker.refresh_fit()
    np.testing.assert_allclose(report.affine[:3, 3], shift, atol=1e-6)
    coarse_picker._apply_affine()
    layer = coarse_picker.viewer.layers["M"]
    np.testing.assert_allclose(layer.scale, (4, 4, 4))
    peak = np.unravel_index(np.argmax(layer.data), layer.data.shape)
    world = np.asarray(peak) * 4 + layer.translate
    np.testing.assert_allclose(world, _coarse_centre((5, 6, 7)) + shift)


def test_preview_maps_moving_layer_through_estimate(coarse_picker, qtbot):
    shift = np.array([8.0, -4.0, 4.0])
    for index in GRID_POINTS:
        p = _coarse_centre(np.asarray(index) % 18)
        _pick(coarse_picker, p + shift, p)
    coarse_picker.refresh_fit()
    coarse_picker.preview_show_window = False
    coarse_picker.preview_size_spin.setValue(10)
    coarse_picker.open_3d_preview(with_estimate=True)
    qtbot.waitUntil(
        lambda: bool(coarse_picker._preview_viewers), timeout=10000
    )
    preview = coarse_picker._preview_viewers[0]
    moving = preview.layers["M"]
    assert moving.data.shape == (10, 10, 10)
    np.testing.assert_allclose(moving.scale, (8, 8, 8))
    np.testing.assert_allclose(
        moving.affine.affine_matrix[:3, 3], shift, atol=1e-6
    )
    picks = preview.layers["Moving picks"].data
    np.testing.assert_allclose(
        picks, preview.layers["Fixed picks"].data, atol=1e-6
    )


def test_preview_uses_provider_when_given(coarse_picker, qtbot):
    calls = []

    def provider(name, max_size):
        calls.append((name, max_size))
        return np.ones((5, 5, 5)), (16, 16, 16), (7.5, 7.5, 7.5)

    coarse_picker.preview_data_provider = provider
    coarse_picker.preview_show_window = False
    coarse_picker.open_3d_preview(with_estimate=False)
    qtbot.waitUntil(
        lambda: bool(coarse_picker._preview_viewers), timeout=10000
    )
    preview = coarse_picker._preview_viewers[0]
    assert {c[0] for c in calls} == {"F", "M"}
    np.testing.assert_allclose(preview.layers["F"].scale, (16, 16, 16))


def test_view_estimator_runs_in_widget(make_napari_viewer, qtbot, tmp_path):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((40, 60, 80), np.float32), name="Fixed")
    viewer.add_image(np.zeros((40, 60, 80), np.float32), name="Moving")
    estimator = partial(
        estimate_affine_from_points_components,
        fix_rotation=(False, True, True),
        fix_scale=(True, False, False),
    )
    widget = PointPickerWidget(
        viewer,
        "Fixed",
        "Moving",
        estimator,
        autosave_dir=None,
        backup_dir=tmp_path / "backup",
    )
    qtbot.addWidget(widget)
    _pick_translation(widget, (2, 3, -1), GRID_POINTS)
    report = widget.refresh_fit()
    assert report.ok and report.rms < 1e-4
    assert "view_transform" not in widget.build_save_payload()["estimator"]
    assert "fix_rotation" in widget.build_save_payload()["estimator"]


def test_hotkeys_dispatch_to_picker_but_not_from_preview(picker, qtbot):
    from app_model.types import KeyBinding
    from napari.components import ViewerModel

    def press(viewer, key):
        ViewerModel.class_keymap[KeyBinding.from_str(key)](viewer)

    picker.viewer.dims.point = (1, 2, 3)
    press(picker.viewer, "Q")
    assert picker.point_pairs[picker.active_pair_id].layer1_coords == (
        1,
        2,
        3,
    )
    press(picker.viewer, "N")
    assert picker.active_pair_id == 2

    picker.preview_show_window = False
    picker.open_3d_preview(with_estimate=False)
    qtbot.waitUntil(lambda: bool(picker._preview_viewers), timeout=10000)
    press(picker._preview_viewers[0], "W")
    assert picker.point_pairs[2].layer2_coords is None


def test_preview_downsampling_keeps_single_voxel_beads():
    from napari_orthogonal_views.point_picker_widget import (
        downsample_for_preview,
    )

    data = np.zeros((9, 9, 9), np.float32)
    data[3, 5, 7] = 1.0
    out, scale, translate = downsample_for_preview(
        data, (1, 1, 1), (0, 0, 0), 3
    )
    assert out.shape == (3, 3, 3) and out[1, 1, 2] == 1.0
    np.testing.assert_allclose(scale, 3)
    # Block 1 covers voxels 3..5, centred on 4.
    np.testing.assert_allclose(
        translate + scale * np.array([1, 1, 2]), (4, 4, 7)
    )


def test_preview_volumes_have_no_nan_or_inf():
    from napari_orthogonal_views.point_picker_widget import (
        downsample_for_preview,
    )

    data = np.ones((4, 4, 4), np.float16)
    data[0, 0, 0] = np.nan
    data[2, 2, 2] = np.inf
    out, _, _ = downsample_for_preview(data, (1, 1, 1), (0, 0, 0), 2)
    assert np.all(np.isfinite(out))
    assert out[1, 1, 1] == np.finfo(np.float16).max


def test_preview_places_moving_volume_through_linear_estimate(
    coarse_picker, qtbot
):
    """World position of a voxel in the preview, not just the stored affine."""
    affine = np.eye(4)
    affine[:3, :3] = [[1, 0.02, 0], [0, 0.97, 0.05], [0, -0.05, 1.02]]
    affine[:3, 3] = [12, -8, 20]
    rng = np.random.default_rng(0)
    for p in rng.uniform(0, 70, (8, 3)):
        coarse_picker.add_pair(record_undo=False)
        pair = coarse_picker.point_pairs[coarse_picker.active_pair_id]
        pair.layer1_coords = tuple(apply_affine(affine, p[None])[0])
        pair.layer2_coords = tuple(p)
    coarse_picker.refresh_fit()
    coarse_picker.preview_show_window = False
    coarse_picker.preview_size_spin.setValue(10)  # stride 2: scale 8
    coarse_picker.open_3d_preview(with_estimate=True)
    qtbot.waitUntil(
        lambda: bool(coarse_picker._preview_viewers), timeout=10000
    )
    moving = coarse_picker._preview_viewers[0].layers["M"]
    peak = np.unravel_index(np.argmax(moving.data), moving.data.shape)
    expected = apply_affine(affine, _coarse_centre((5, 6, 7))[None])[0]
    # Downsampled 2x by block maximum: the bead is kept, at the centre of
    # its 2x2x2 block, within half a coarse block (4 world units) of it.
    assert moving.data.max() == 1.0
    np.testing.assert_allclose(
        np.asarray(moving.data_to_world(peak)), expected, atol=4.5
    )
