#!/usr/bin/env python
"""Pick projects: one folder per bead store, holding its picks and results.

Every run of ``pick_impp_views.py`` opens (or creates) the
folder for its store under ``picks/`` and starts a session in it. A session
keeps the picks (autosaved after every edit), what they were picked on, and,
after each **Apply**, a ``result.yaml`` with the fit and the calibration
entries it changes. ``assemble`` writes a complete calibration file from the
base calibration plus the sessions you accepted, which impp reads directly
(``Project.create(calibration=path)``) or which can be filed into its
library. Nothing here holds image data.

Layout::

    picks/<store>/store.yaml                    what the store is
    picks/<store>/sessions/<session>/session.yaml   how the session was run
    picks/<store>/sessions/<session>/picks.json     the picks
    picks/<store>/sessions/<session>/result.yaml    fit and calibration entries
    picks/<store>/calibration/<base>_picked.yaml    assembled calibration
    picks/<store>/calibration/assembly.yaml         what went into it

Usage::

    python pick_projects.py list [STORE]
    python pick_projects.py accept SESSION [--undo]
    python pick_projects.py assemble STORE [--base CALIBRATION] [--force]

``STORE`` is a store folder under ``picks/`` or the store's own path (any
pyramid level of it); ``SESSION`` is a session folder.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PICKS_ROOT = Path(__file__).resolve().parent / "picks"
RESULT_VERSION = 1

# View 2's y translation beyond this many level-0 rows is the beads'
# top/bottom spacing, not calibration: it is set per sample, and left in the
# matrices it cuts that many rows of the top view at deskew. Assembly takes
# it out of views 2 and 3 (impp does the same at load; doing it here keeps
# the filed matrices safe on their own).
TOP_Y_SHIFT_MIN = 10
TOP_VIEWS = (2, 3)


# ------------------------------------------------------------------- YAML


class _Dumper(yaml.SafeDumper):
    """Matrices as one flow-style row per line, and no YAML aliases."""

    def ignore_aliases(self, data):
        return True


def _represent_list(dumper, data):
    flow = bool(data) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in data
    )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=flow
    )


_Dumper.add_representer(list, _represent_list)


def to_plain(value):
    """numpy arrays and scalars to lists and Python numbers, recursively."""
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def write_yaml(path: Path, data: dict, header: str | None = None) -> None:
    text = yaml.dump(
        to_plain(data), Dumper=_Dumper, sort_keys=False, width=100
    )
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def repo_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PICKS_ROOT.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


# ----------------------------------------------------------------- stores


def _zarr_meta(path: Path) -> tuple[str | None, dict]:
    """``("array" | "group" | None, attributes)`` for zarr v2 or v3."""
    path = Path(path)
    if (path / ".zarray").exists():
        attrs_path = path / ".zattrs"
        attrs = (
            json.loads(attrs_path.read_text()) if attrs_path.exists() else {}
        )
        return "array", attrs
    if (path / "zarr.json").exists():
        meta = json.loads((path / "zarr.json").read_text())
        attrs = meta.get("attributes") or {}
        return meta.get("node_type"), attrs.get("ome", attrs)
    if (path / ".zattrs").exists():
        return "group", json.loads((path / ".zattrs").read_text())
    return None, {}


def _array_shape(path: Path) -> list | None:
    path = Path(path)
    for name in (".zarray", "zarr.json"):
        if (path / name).exists():
            return json.loads((path / name).read_text()).get("shape")
    return None


def store_root(zarr_path: Path) -> Path:
    """The store a pyramid-level array belongs to (the multiscale group)."""
    path = Path(zarr_path).expanduser().resolve()
    kind, _ = _zarr_meta(path)
    _, parent_attrs = _zarr_meta(path.parent)
    if kind == "array" and "multiscales" in parent_attrs:
        return path.parent
    return path


def store_slug(root: Path) -> str:
    """Readable, collision-safe folder name for a store path."""
    root = Path(root)
    readable = f"{root.parent.name}__{root.name.removesuffix('.zarr')}"
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", readable).strip("-")[:80]
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:6]
    return f"{readable}-{digest}"


@dataclass
class StoreProject:
    dir: Path
    root: Path

    @classmethod
    def for_store(
        cls, zarr_path: Path, picks_root: Path = PICKS_ROOT
    ) -> StoreProject:
        root = store_root(zarr_path)
        project = cls(Path(picks_root) / store_slug(root), root)
        project.ensure()
        return project

    @classmethod
    def resolve(cls, value: str | Path, picks_root: Path = PICKS_ROOT):
        """A store folder under ``picks/``, or the store's own path."""
        path = Path(value).expanduser()
        if (path / "store.yaml").exists():
            info = read_yaml(path / "store.yaml")
            return cls(path, Path(info["store_path"]))
        root = store_root(path)
        folder = Path(picks_root) / store_slug(root)
        if not (folder / "store.yaml").exists():
            raise SystemExit(f"No pick project for {root} under {picks_root}")
        return cls(folder, root)

    def ensure(self) -> None:
        info_path = self.dir / "store.yaml"
        if info_path.exists():
            return
        write_yaml(
            info_path,
            {
                "store_path": str(self.root),
                "created_at": now(),
                **_describe_store(self.root),
            },
            header="A bead store this folder holds picks for (no image data).",
        )

    @property
    def sessions_dir(self) -> Path:
        return self.dir / "sessions"

    def new_session(self, label: str, info: dict) -> Session:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
        path = self.sessions_dir / f"{stamp}_{label}"
        suffix = 1
        while path.exists():
            suffix += 1
            path = self.sessions_dir / f"{stamp}_{label}-{suffix}"
        session = Session(path)
        write_yaml(
            session.info_path,
            {"created_at": now(), "repo_commit": repo_commit(), **info},
            header="How this picking session was run.",
        )
        return session

    def sessions(self) -> list[Session]:
        if not self.sessions_dir.exists():
            return []
        return [Session(p) for p in sorted(self.sessions_dir.iterdir())]


def _describe_store(root: Path) -> dict:
    """Shapes per pyramid level and channel labels, from the metadata."""
    info: dict[str, Any] = {}
    _, attrs = _zarr_meta(root)
    levels = []
    for dataset in (attrs.get("multiscales") or [{}])[0].get("datasets", []):
        shape = _array_shape(root / str(dataset["path"]))
        if shape is not None:
            levels.append({"path": str(dataset["path"]), "shape": shape})
    if levels:
        info["levels"] = levels
    channels = attrs.get("omero", {}).get("channels", [])
    if channels:
        info["channels"] = [c.get("label") for c in channels]
    return info


# --------------------------------------------------------------- sessions


@dataclass
class Session:
    dir: Path

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def info_path(self) -> Path:
        return self.dir / "session.yaml"

    @property
    def picks_path(self) -> Path:
        return self.dir / "picks.json"

    @property
    def result_path(self) -> Path:
        return self.dir / "result.yaml"

    def info(self) -> dict:
        return read_yaml(self.info_path) if self.info_path.exists() else {}

    def result(self) -> dict | None:
        if not self.result_path.exists():
            return None
        return read_yaml(self.result_path)

    def write_result(self, result: dict) -> Path:
        """Write ``result.yaml``; a new result starts unaccepted."""
        payload = {
            "format_version": RESULT_VERSION,
            "written_at": now(),
            "accepted": False,
            **result,
        }
        write_yaml(
            self.result_path,
            payload,
            header=(
                "The latest applied fit of this session and the calibration\n"
                "entries it changes. Accept it with\n"
                f"  python pick_projects.py accept {self.dir}\n"
                "then build the calibration file with "
                "`python pick_projects.py assemble`."
            ),
        )
        return self.result_path

    def remove_if_empty(self) -> bool:
        """Delete a session that never got picks or a result."""
        files = (
            {p.name for p in self.dir.iterdir()}
            if self.dir.exists()
            else set()
        )
        if files - {"session.yaml"}:
            return False
        if self.info_path.exists():
            self.info_path.unlink()
        if self.dir.exists():
            self.dir.rmdir()
        return True

    def set_accepted(self, accepted: bool) -> None:
        result = self.result()
        if result is None:
            raise SystemExit(f"{self.dir} has no result.yaml to accept.")
        result["accepted"] = accepted
        result["accepted_at" if accepted else "unaccepted_at"] = now()
        write_yaml(self.result_path, result)


def calibration_entry_value(calibration: dict, kind: str, key: tuple):
    """The value a calibration holds for one entry, or ``None``."""
    if kind == "view_transforms":
        value = (calibration.get("view_transforms") or {}).get(key[0])
        if value is None:
            value = (calibration.get("view_transforms") or {}).get(str(key[0]))
        return value
    transforms = (calibration.get("camera_alignment") or {}).get(
        "transforms"
    ) or {}
    return (transforms.get(key[0]) or {}).get(key[1])


def iter_entries(entries: dict):
    """``(kind, key, entry)`` for every calibration entry in a result."""
    for view, entry in (entries.get("view_transforms") or {}).items():
        yield "view_transforms", (int(view),), entry
    for camera, views in (entries.get("camera_alignment") or {}).items():
        for view, entry in views.items():
            yield "camera_alignment", (camera, view), entry


def describe_key(kind: str, key: tuple) -> str:
    if kind == "view_transforms":
        return f"view_transforms[{key[0]}]"
    return f"camera_alignment.transforms.{key[0]}.{key[1]}"


def snapshot_base_values(entries: dict, base_path: Path | None) -> dict:
    """Record what the base calibration held for each entry being changed.

    Assembly compares these with the base file as it is then, so a result
    computed against a base that has since changed is caught.
    """
    if base_path is None:
        return entries
    base = read_yaml(base_path)
    entries = copy.deepcopy(entries)
    for kind, key, entry in iter_entries(entries):
        entry["base_value"] = calibration_entry_value(base, kind, key)
    return entries


# --------------------------------------------------------------- assemble


def _same_value(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b)


def remove_top_bottom_y(view_transforms: dict) -> dict[int, int]:
    """Take the beads' top/bottom spacing out of views 2 and 3's y.

    The spacing is view 2's whole-row y translation when it is beyond
    ``TOP_Y_SHIFT_MIN``. It is removed from view 2, and from view 3 only
    when view 3 carries it too (its own y is beyond the threshold with the
    same sign), so view 3 keeps its difference from view 2. A view 3 that
    was never given the spacing is left alone. Returns the rows removed per
    view.
    """
    matrix = view_transforms.get(2)
    if matrix is None:
        return {}
    shift = int(round(float(matrix[1][3])))
    if abs(shift) <= TOP_Y_SHIFT_MIN:
        return {}
    removed = {}
    for view in TOP_VIEWS:
        if view not in view_transforms:
            continue
        m = np.array(view_transforms[view], dtype=float)
        carries = abs(m[1, 3]) > TOP_Y_SHIFT_MIN and np.sign(m[1, 3]) == (
            np.sign(shift)
        )
        if view == 2 or carries:
            m[1, 3] -= shift
            view_transforms[view] = m.tolist()
            removed[view] = shift
    return removed


def assemble(
    project: StoreProject,
    base_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Write the base calibration with every accepted session's entries."""
    accepted = []
    for session in project.sessions():
        result = session.result()
        if result and result.get("accepted"):
            accepted.append((session, result))
    if not accepted:
        raise SystemExit(
            f"No accepted sessions in {project.dir}. Accept one with "
            "`python pick_projects.py accept SESSION`."
        )

    recorded_bases = {
        (r.get("base_calibration") or {}).get("path") for _, r in accepted
    } - {None}
    if base_path is None:
        if len(recorded_bases) != 1:
            raise SystemExit(
                "Pass --base: the accepted sessions name "
                f"{len(recorded_bases)} different base calibrations "
                f"({sorted(recorded_bases)})."
            )
        base_path = Path(recorded_bases.pop())
    base_path = Path(base_path).expanduser().resolve()
    base = read_yaml(base_path)
    base_sha = file_sha256(base_path)
    calibration = copy.deepcopy(base)
    view_transforms = {
        int(k): v
        for k, v in (calibration.get("view_transforms") or {}).items()
    }

    applied = []
    problems = []
    claimed: dict[str, str] = {}
    for session, result in accepted:
        recorded = result.get("base_calibration") or {}
        same_base = recorded.get("sha256") == base_sha
        for kind, key, entry in iter_entries(
            result.get("calibration_entries") or {}
        ):
            label = describe_key(kind, key)
            current = calibration_entry_value(base, kind, key)
            if not same_base and not _same_value(
                entry.get("base_value"), current
            ):
                problems.append(
                    f"{session.name}: {label} was solved against a base "
                    "value that differs from the base calibration's now"
                )
                continue
            a_old = (entry.get("solved_from") or {}).get("a_old")
            output_inverse = (entry.get("solved_from") or {}).get(
                "output_inverse"
            )
            if (
                kind == "view_transforms"
                and a_old is not None
                and not output_inverse
                and current is not None
                and not _same_value(a_old, current)
            ):
                problems.append(
                    f"{session.name}: {label} was solved against an A_old "
                    f"({(entry.get('solved_from') or {}).get('a_old_source')})"
                    " that differs from the base calibration's entry; use "
                    "the calibration the store was processed with as base"
                )
                if not force:
                    continue
            if label in claimed:
                problems.append(
                    f"{session.name}: {label} is also set by "
                    f"{claimed[label]}; the later session wins"
                )
            claimed[label] = session.name
            if kind == "view_transforms":
                view_transforms[key[0]] = entry["matrix"]
            else:
                camera_alignment = calibration.setdefault(
                    "camera_alignment", {}
                )
                transforms = camera_alignment.setdefault("transforms", {})
                transforms.setdefault(key[0], {})[key[1]] = entry["matrix"]
            applied.append({"session": session.name, "entry": label})

    stale = [p for p in problems if "differ" in p]
    if stale and not force:
        raise SystemExit(
            "Refusing to assemble (use --force to skip these entries):\n  "
            + "\n  ".join(stale)
        )

    removed = remove_top_bottom_y(view_transforms)
    if view_transforms:
        calibration["view_transforms"] = dict(sorted(view_transforms.items()))

    out_dir = project.dir / "calibration"
    out_path = out_dir / f"{base_path.stem}_picked.yaml"
    header = (
        f"Assembled by pick_projects.py on {now()}.\n"
        f"Base: {base_path} (sha256 {base_sha[:16]})\n"
        "Sessions: "
        + ", ".join(s.name for s, _ in accepted)
        + "\n"
        + (
            "Removed the beads' top/bottom spacing from the y translation: "
            + ", ".join(f"view {v}: {r} rows" for v, r in removed.items())
            + ".\n"
            if removed
            else ""
        )
        + "See assembly.yaml in this folder for every entry and its source."
    )
    write_yaml(out_path, calibration, header=header)
    write_yaml(
        out_dir / "assembly.yaml",
        {
            "assembled_at": now(),
            "output": out_path.name,
            "base_calibration": {"path": str(base_path), "sha256": base_sha},
            "entries": applied,
            "top_bottom_y_removed_rows": removed,
            "problems": problems,
            "acquisition_changes": {
                s.name: r["acquisition_changes"]
                for s, r in accepted
                if r.get("acquisition_changes")
            },
        },
        header="What went into the assembled calibration file.",
    )
    return out_path


# -------------------------------------------------------------------- CLI


def _cmd_list(args) -> None:
    if args.store:
        projects = [StoreProject.resolve(args.store)]
    else:
        projects = [
            StoreProject.resolve(info.parent)
            for info in sorted(PICKS_ROOT.glob("*/store.yaml"))
        ]
    for project in projects:
        print(f"{project.dir.name}  ({project.root})")
        for session in project.sessions():
            result = session.result()
            if result is None:
                status = "no result"
            else:
                fit = result.get("fit") or {}
                rms = fit.get("rms")
                entries = [
                    describe_key(kind, key)
                    for kind, key, _ in iter_entries(
                        result.get("calibration_entries") or {}
                    )
                ]
                status = (
                    ("ACCEPTED" if result.get("accepted") else "candidate")
                    + (f", RMS {rms:.2f}" if rms is not None else "")
                    + (f", sets {', '.join(entries)}" if entries else "")
                )
            print(f"  {session.name}: {status}")


def _cmd_accept(args) -> None:
    session = Session(Path(args.session).expanduser())
    session.set_accepted(not args.undo)
    print(f"{'Unaccepted' if args.undo else 'Accepted'} {session.name}.")


def _cmd_assemble(args) -> None:
    project = StoreProject.resolve(args.store)
    out = assemble(project, args.base, force=args.force)
    print(f"Wrote {out}")
    summary = read_yaml(out.parent / "assembly.yaml")
    for problem in summary.get("problems") or []:
        print(f"  note: {problem}")
    for view, rows in (summary.get("top_bottom_y_removed_rows") or {}).items():
        print(
            f"  removed {rows} rows of top/bottom y translation from view "
            f"{view}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="Sessions and their results.")
    p.add_argument("store", nargs="?")
    p.set_defaults(func=_cmd_list)
    p = sub.add_parser("accept", help="Mark a session's result as accepted.")
    p.add_argument("session")
    p.add_argument("--undo", action="store_true")
    p.set_defaults(func=_cmd_accept)
    p = sub.add_parser(
        "assemble", help="Write the calibration file from accepted sessions."
    )
    p.add_argument("store")
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_assemble)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
