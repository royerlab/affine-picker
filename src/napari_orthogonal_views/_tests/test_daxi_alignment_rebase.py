import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

sys.path.insert(0, str(Path(__file__).parents[3]))
import load_zarr_views_for_registration as registration  # noqa: E402


def _write_project(
    path: Path,
    *,
    output_path: Path | None = None,
    top_bottom_offsets: tuple[int, int, int] = (0, 0, 0),
    pair_0_1_offset: tuple[int, int, int] = (0, 0, 0),
    pair_2_3_offset: tuple[int, int, int] = (0, 0, 0),
    post_deskew_flip_axes: tuple[int, ...] = (),
) -> None:
    identity = np.eye(4).tolist()
    settings = {
        "alignment": {
            "top_bottom_offsets": list(top_bottom_offsets),
            "view_pair_offsets": {
                "pair_0_1": list(pair_0_1_offset),
                "pair_2_3": list(pair_2_3_offset),
            },
        },
        "processing": {
            "camera_axes_to_flip_post_deskew": {
                0: list(post_deskew_flip_axes)
            },
            "view_transforms": {str(view): identity for view in range(4)},
        },
    }
    if output_path is not None:
        settings["job"] = {"output_path": str(output_path)}
    path.mkdir()
    with (path / "impp_settings.yaml").open("w") as file:
        yaml.safe_dump(settings, file)


def test_current_shape_is_inferred_from_project_output(tmp_path):
    output_path = tmp_path / "deskewed.zarr"
    zarr.create_array(
        output_path / "0",
        shape=(1, 4, 10, 20, 30),
        chunks=(1, 1, 5, 20, 30),
        dtype="f2",
        zarr_format=2,
    )
    project = tmp_path / "project"
    _write_project(project, output_path=output_path)

    shape, source = registration._resolve_current_spatial_shape(
        current_shape=None,
        current_zarr_path=None,
        project_folder=project,
        points_zarr_path=tmp_path / "old.zarr" / "0",
    )

    assert shape == (10, 20, 30)
    assert str(output_path / "0") in source


def test_rebased_mode_does_not_fall_back_to_old_shape(tmp_path):
    with pytest.raises(
        SystemExit, match="needs the regenerated/current canvas shape"
    ):
        registration._resolve_current_spatial_shape(
            current_shape=None,
            current_zarr_path=None,
            project_folder=None,
            points_zarr_path=tmp_path / "old.zarr" / "0",
        )


def test_changed_bottom_flip_extent_is_included_in_rebase(tmp_path):
    old_project = tmp_path / "old"
    current_project = tmp_path / "current"
    _write_project(old_project)
    _write_project(current_project)

    old_slot = registration._infer_pipeline_slot(
        alignment_mode="view",
        project_folder=old_project,
        transform_yaml=None,
        view_id=2,
        camera_id=0,
        view_shape_zyx=(10, 24, 30),
        allow_identity_old_transform=False,
    )
    current_slot = registration._infer_pipeline_slot(
        alignment_mode="view",
        project_folder=current_project,
        transform_yaml=None,
        view_id=2,
        camera_id=0,
        view_shape_zyx=(10, 20, 30),
        allow_identity_old_transform=False,
    )

    rebase = registration._full_pipeline_affine(current_slot) @ np.linalg.inv(
        registration._full_pipeline_affine(old_slot)
    )
    point_old = np.array([2.0, 7.0, 8.0, 1.0])

    # The raw point represented by old displayed Y=7 is displayed at Y=3
    # after the bottom-view flip extent shrinks from 24 to 20.
    np.testing.assert_allclose(rebase @ point_old, [2.0, 3.0, 8.0, 1.0])


def test_bottom_flip_uses_shape_before_top_bottom_canvas_growth(tmp_path):
    project = tmp_path / "project"
    _write_project(project, top_bottom_offsets=(5, 190, -7))

    slot = registration._infer_pipeline_slot(
        alignment_mode="view",
        project_folder=project,
        transform_yaml=None,
        view_id=2,
        camera_id=0,
        # The saved Zarr canvas grew from (10, 24, 30) to (15, 214, 30).
        view_shape_zyx=(15, 214, 30),
        allow_identity_old_transform=False,
    )

    # The pipeline flips Y around the pre-growth height 24 and then shifts by
    # 190: y_out = -(y_in) + 23 + 190.  Using final height 214 would produce
    # the erroneous translation 403.
    expected = np.eye(4)
    expected[1, 1] = -1.0
    expected[:3, 3] = [5.0, 213.0, -7.0]
    np.testing.assert_allclose(slot.post_processing_affine, expected)
    assert "pre-placement shape=(10, 24, 30)" in slot.post_source


def test_bottom_pair_offset_is_applied_before_top_bottom_flip(tmp_path):
    project = tmp_path / "project"
    _write_project(
        project,
        top_bottom_offsets=(5, 190, -7),
        pair_0_1_offset=(3, 12, 240),
        pair_2_3_offset=(3, 12, 240),
        post_deskew_flip_axes=(2,),
    )

    top_slot = registration._infer_pipeline_slot(
        alignment_mode="view",
        project_folder=project,
        transform_yaml=None,
        view_id=1,
        camera_id=0,
        view_shape_zyx=(15, 214, 30),
        allow_identity_old_transform=False,
    )
    bottom_slot = registration._infer_pipeline_slot(
        alignment_mode="view",
        project_folder=project,
        transform_yaml=None,
        view_id=3,
        camera_id=0,
        view_shape_zyx=(15, 214, 30),
        allow_identity_old_transform=False,
    )

    # Top views have no later top/bottom flip, so their configured pair
    # offset is already expressed in output coordinates.
    np.testing.assert_allclose(
        top_slot.pair_offset_affine[:3, 3],
        [3.0, 12.0, 240.0],
    )

    # The pipeline shifts view 3 before its top/bottom Y-flip. In the final
    # output frame, the configured +12 Y shift is therefore -12. This was
    # previously modeled as +12 and caused a 24-pixel relative displacement
    # after pair offsets were zeroed and absorbed into view transforms.
    np.testing.assert_allclose(
        bottom_slot.pair_offset_affine[:3, 3],
        [3.0, -12.0, 240.0],
    )

    pre_shape = (10, 24, 30)
    post_flip = registration._flip_affine_for_axes(pre_shape, (2,))
    pair_offset = registration._translation_affine((3, 12, 240))
    placement = registration._translation_affine(
        (5, 190, -7)
    ) @ registration._flip_affine_for_axes(pre_shape, (1,))
    expected_post = placement @ pair_offset @ post_flip
    np.testing.assert_allclose(bottom_slot.post_affine, expected_post)


def test_top_bottom_canvas_growth_matches_per_camera_pipeline_rule():
    alignment = {
        "per_camera_offsets": True,
        "camera_offsets": {
            "camera_0": {"top_bottom_offsets": [5, -4, 11]},
            "camera_1": {"top_bottom_offsets": [2, 9, -3]},
        },
    }

    assert registration._top_bottom_canvas_growth(alignment) == (5, 9, 11)


def _write_pyramid(root: Path, shapes, scales) -> None:
    import json

    root.mkdir()
    datasets = []
    for level, (shape, scale) in enumerate(zip(shapes, scales, strict=True)):
        zarr.open(
            str(root / str(level)), mode="w", shape=shape, dtype="uint16"
        )
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, *scale]}
                ],
            }
        )
    (root / ".zattrs").write_text(
        json.dumps({"multiscales": [{"version": "0.4", "datasets": datasets}]})
    )


def test_pyramid_level_factor_and_full_resolution_shape(tmp_path):
    root = tmp_path / "deskewed.zarr"
    _write_pyramid(
        root,
        [(1, 2, 40, 80, 120), (1, 2, 20, 40, 60), (1, 2, 10, 20, 30)],
        [(1.2, 0.4, 0.4), (2.4, 0.8, 0.8), (4.8, 1.6, 1.6)],
    )
    level = registration._pyramid_level(root / "2")
    assert level.factor == (4.0, 4.0, 4.0)
    assert not level.is_full_resolution
    assert registration._full_resolution_shape(level, (10, 20, 30)) == (
        40,
        80,
        120,
    )
    assert registration._pyramid_level(root / "0").is_full_resolution


def test_array_outside_pyramid_is_full_resolution(tmp_path):
    zarr.open(str(tmp_path / "plain"), mode="w", shape=(4, 4), dtype="u1")
    assert registration._pyramid_level(tmp_path / "plain").is_full_resolution


def test_preview_loader_picks_finest_level_that_fits(tmp_path):
    root = tmp_path / "deskewed.zarr"
    _write_pyramid(
        root,
        [(1, 2, 40, 80, 120), (1, 2, 20, 40, 60), (1, 2, 10, 20, 30)],
        [(1, 1, 1), (2, 2, 2), (4, 4, 4)],
    )
    level = registration._pyramid_level(root / "0")
    load = registration._preview_level_loader(
        {"F": (level, 1, registration.ViewFlips())}, timepoint=0
    )
    volume, scale, translate = load("F", 64)
    assert volume.shape == (20, 40, 60)
    assert scale == (2.0, 2.0, 2.0)
    assert translate == (0.5, 0.5, 0.5)
    assert load("unknown layer", 64) is None


def test_full_resolution_picks_read_current_shape_at_level_0(tmp_path):
    root = tmp_path / "current.zarr"
    _write_pyramid(
        root,
        [(1, 2, 40, 80, 120), (1, 2, 10, 20, 30)],
        [(1, 1, 1), (4, 4, 4)],
    )
    shape, _ = registration._resolve_current_spatial_shape(
        current_shape=None,
        current_zarr_path=root / "1",
        project_folder=None,
        points_zarr_path=tmp_path / "old.zarr" / "1",
        full_resolution=True,
    )
    assert shape == (40, 80, 120)


def test_camera_alignment_result_matches_documented_recipe():
    """The worked example in TRANSFORMS_YAML_HOWTO.md (camera mode)."""
    import argparse

    world = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.9942680048, -0.0104417158, 11.3612280404],
            [0.0, 0.009476072, 0.9945267206, 3.1706529672],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected = [
        [1.0056644090, 0.0105586524, -11.4590605055],
        [-0.0095821944, 1.0054027959, -3.0789178625],
        [0.0, 0.0, 1.0],
    ]
    args = argparse.Namespace(
        camera_alignment_key="camera_1", camera_alignment_views=[0, 1]
    )
    result = registration._camera_alignment_result(world, args, 0)
    entries = result["calibration_entries"]["camera_alignment"]["camera_1"]
    assert set(entries) == {"view_0", "view_1"}
    np.testing.assert_allclose(entries["view_0"]["matrix"], expected, atol=1e-9)

    args.camera_alignment_key = None
    unkeyed = registration._camera_alignment_result(world, args, 0)
    assert unkeyed["calibration_entries"] == {}
    assert "view_0" in unkeyed["unkeyed_camera_alignment"]


def test_level_factor_comes_from_shapes_when_scale_metadata_is_wrong(
    tmp_path, capsys
):
    """A camera store whose level 1 declares scale / 4 instead of * 4."""
    root = tmp_path / "pos0"
    _write_pyramid(
        root,
        [(1, 1, 804, 1300, 2048), (1, 1, 201, 325, 512)],
        [(1.24427, 0.4394, 0.4394), (0.3110675, 0.10985, 0.10985)],
    )
    level = registration._pyramid_level(root / "1")
    assert level.factor == (4.0, 4.0, 4.0)
    assert "metadata is wrong" in capsys.readouterr().out
