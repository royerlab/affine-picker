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
) -> None:
    identity = np.eye(4).tolist()
    settings = {
        "alignment": {
            "top_bottom_offsets": list(top_bottom_offsets),
            "view_pair_offsets": {
                "pair_0_1": [0, 0, 0],
                "pair_2_3": [0, 0, 0],
            },
        },
        "processing": {
            "camera_axes_to_flip_post_deskew": {0: []},
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


def test_top_bottom_canvas_growth_matches_per_camera_pipeline_rule():
    alignment = {
        "per_camera_offsets": True,
        "camera_offsets": {
            "camera_0": {"top_bottom_offsets": [5, -4, 11]},
            "camera_1": {"top_bottom_offsets": [2, 9, -3]},
        },
    }

    assert registration._top_bottom_canvas_growth(alignment) == (5, 9, 11)
