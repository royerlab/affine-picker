import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

sys.path.insert(0, str(Path(__file__).parents[3]))
import pick_projects as pp  # noqa: E402


def _matrix(y=0.0, x=0.0):
    m = np.eye(4)
    m[1, 3] = y
    m[2, 3] = x
    return m.tolist()


def _base(tmp_path, view_y=(0.0, 0.0, 0.0, 0.0)) -> Path:
    identity3 = np.eye(3).tolist()
    data = {
        "microscope": "daxi-a",
        "date_start": "2026-01-01",
        "date_end": "2027-01-01",
        "view_transforms": {v: _matrix(y) for v, y in enumerate(view_y)},
        "camera_alignment": {
            "reference_camera": 0,
            "transforms": {
                "camera_1": {f"view_{v}": identity3 for v in range(4)}
            },
        },
    }
    path = tmp_path / "daxi-a_2026-01-01.yaml"
    # Shared lists, as the real files write them with YAML anchors.
    path.write_text(yaml.safe_dump(data))
    return path


def _store(tmp_path) -> pp.StoreProject:
    root = tmp_path / "data" / "deskewed.zarr"
    zarr.open(str(root / "0"), mode="w", shape=(1, 1, 4, 4, 4), dtype="u1")
    (root / ".zattrs").write_text(
        json.dumps({"multiscales": [{"datasets": [{"path": "0"}]}]})
    )
    return pp.StoreProject.for_store(root / "0", tmp_path / "picks")


def _session(project, base, entries, label="s"):
    session = project.new_session(label, {"note": "test"})
    session.picks_path.write_text("{}")
    entries = pp.snapshot_base_values(entries, base)
    session.write_result(
        {
            "calibration_entries": entries,
            "base_calibration": {
                "path": str(base),
                "sha256": pp.file_sha256(base),
            },
        }
    )
    return session


def _view_entry(view, matrix, a_old):
    return {
        "view_transforms": {
            view: {"matrix": matrix, "solved_from": {"a_old": a_old}}
        }
    }


def test_store_folder_is_shared_by_every_pyramid_level(tmp_path):
    project = _store(tmp_path)
    assert project.root.name == "deskewed.zarr"
    assert project.dir.name.startswith("data__deskewed-")
    again = pp.StoreProject.resolve(project.root / "0", tmp_path / "picks")
    assert again.dir == project.dir
    assert yaml.safe_load((project.dir / "store.yaml").read_text())[
        "levels"
    ] == [{"path": "0", "shape": [1, 1, 4, 4, 4]}]


def test_empty_session_is_removed(tmp_path):
    project = _store(tmp_path)
    session = project.new_session("x", {})
    assert session.remove_if_empty()
    assert not session.dir.exists()
    kept = project.new_session("y", {})
    kept.picks_path.write_text("{}")
    assert not kept.remove_if_empty()


def test_assemble_writes_accepted_entries_only(tmp_path):
    base = _base(tmp_path)
    project = _store(tmp_path)
    new = _matrix(y=3.0, x=2.0)
    accepted = _session(project, base, _view_entry(1, new, _matrix()))
    _session(project, base, _view_entry(0, _matrix(x=9.0), _matrix()))
    accepted.set_accepted(True)
    out = pp.assemble(project)
    result = yaml.safe_load(out.read_text())
    assert result["view_transforms"][1] == new
    assert result["view_transforms"][0] == _matrix()  # not accepted
    assert result["microscope"] == "daxi-a"


def test_assemble_requires_an_accepted_session(tmp_path):
    base = _base(tmp_path)
    project = _store(tmp_path)
    _session(project, base, _view_entry(1, _matrix(), _matrix()))
    with pytest.raises(SystemExit, match="No accepted sessions"):
        pp.assemble(project)


def test_top_bottom_y_is_removed_from_views_carrying_it(tmp_path):
    base = _base(tmp_path, view_y=(0.0, 0.0, -230.0, -255.0))
    project = _store(tmp_path)
    session = _session(
        project, base, _view_entry(2, _matrix(y=-231.4), _matrix(y=-230.0))
    )
    session.set_accepted(True)
    out = pp.assemble(project)
    vt = yaml.safe_load(out.read_text())["view_transforms"]
    assert vt[2][1][3] == pytest.approx(-0.4)
    assert vt[3][1][3] == pytest.approx(-24.0)  # keeps its offset from 2
    summary = yaml.safe_load((out.parent / "assembly.yaml").read_text())
    assert summary["top_bottom_y_removed_rows"] == {2: -231, 3: -231}


def test_view_3_without_the_spacing_is_left_alone(tmp_path):
    base = _base(tmp_path, view_y=(0.0, 0.0, 1.0, -1.0))
    project = _store(tmp_path)
    session = _session(
        project, base, _view_entry(2, _matrix(y=-175.0), _matrix(y=1.0))
    )
    session.set_accepted(True)
    vt = yaml.safe_load(pp.assemble(project).read_text())["view_transforms"]
    assert vt[2][1][3] == pytest.approx(0.0)
    assert vt[3][1][3] == pytest.approx(-1.0)


def test_result_solved_against_other_a_old_is_refused(tmp_path):
    base = _base(tmp_path)
    project = _store(tmp_path)
    session = _session(
        project, base, _view_entry(1, _matrix(y=3.0), _matrix(y=50.0))
    )
    session.set_accepted(True)
    with pytest.raises(SystemExit, match="A_old"):
        pp.assemble(project)
    assert pp.assemble(project, force=True).exists()


def test_result_against_a_changed_base_is_refused(tmp_path):
    base = _base(tmp_path)
    project = _store(tmp_path)
    session = _session(project, base, _view_entry(1, _matrix(y=3.0), None))
    session.set_accepted(True)
    data = yaml.safe_load(base.read_text())
    data["view_transforms"][1] = _matrix(y=7.0)
    base.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit, match="differs"):
        pp.assemble(project)


def test_camera_alignment_entry_replaces_one_view_only(tmp_path):
    base = _base(tmp_path)
    project = _store(tmp_path)
    new = [[1.01, 0.0, -7.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]]
    entries = {
        "camera_alignment": {
            "camera_1": {"view_0": {"matrix": new, "solved_from": {}}}
        }
    }
    _session(project, base, entries).set_accepted(True)
    out = pp.assemble(project)
    transforms = yaml.safe_load(out.read_text())["camera_alignment"][
        "transforms"
    ]["camera_1"]
    assert transforms["view_0"] == new
    assert transforms["view_1"] == np.eye(3).tolist()
    assert "&" not in out.read_text()  # no YAML aliases


def test_cli_list_and_accept(tmp_path, capsys, monkeypatch):
    base = _base(tmp_path)
    project = _store(tmp_path)
    session = _session(project, base, _view_entry(1, _matrix(), _matrix()))
    monkeypatch.setattr(pp, "PICKS_ROOT", tmp_path / "picks")
    pp.main(["accept", str(session.dir)])
    pp.main(["list"])
    out = capsys.readouterr().out
    assert "ACCEPTED" in out and "view_transforms[1]" in out
