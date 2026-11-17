#!/usr/bin/env python
"""Pick matching beads in two views of an impp project and turn the fit into calibration entries.

Runs in impp's environment (it imports ``impp``), on views the project has deskewed and stored at level 0::

    cd <impp checkout>
    pixi run python -m impp deskew PROJECT 488nm@0 488nm@1 --t 0 --level 0        # minutes per view
    pixi run python external/napari-orthogonal-views/pick_impp_views.py PROJECT 488nm@0 488nm@1 --t 0

A is fixed and B moves. B is drawn where the project's alignment puts it against A, so world coordinates are A's stored
level-0 voxels, and the solve (``impp_frames``) writes matrices in the pinned calibration file's own frame:

- two views of one objective (``488nm@0 488nm@1``): ``shared-space``, half the correction to each view (default), or
  ``slot``, all of it to B
- one view of each objective (``488nm@0 488nm@2``): ``top-bottom``, B and its objective partner both move
- one view on two cameras (``488nm@0 561nm@0``): ``camera``, a (Y, X) camera alignment for B's camera, written to
  both views of B's objective unless ``--camera-views`` says otherwise

Each run is a session of the bead store's pick project (``picks/``, see ``pick_projects.py``); every **Apply** writes the
session's ``result.yaml``. Accept the sessions you trust and assemble one calibration file::

    python pick_projects.py accept picks/<store>/sessions/<session>
    python pick_projects.py assemble picks/<store>

then pin it in a new project: ``impp.Project.create(RAW, DIR, calibration=FILE, reason=...)``.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE / "src").exists():
    sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

import impp_frames as frames  # noqa: E402
import pick_projects  # noqa: E402
from napari_orthogonal_views import (  # noqa: E402
    downsampled_layer_kwargs,
    estimate_affine_from_points,
    estimate_affine_from_points_components,
    show_point_picker,
)

MODES = ("auto", "shared-space", "slot", "top-bottom", "camera")


# ------------------------------------------------------------------ impp

def _impp():
    try:
        import impp
    except ImportError as error:
        raise SystemExit("pick_impp_views.py imports impp: run it in impp's environment "
                         "(pixi run python external/napari-orthogonal-views/pick_impp_views.py ...)") from error
    return impp


def _pinned_calibration(project):
    from impp.calibration import load_calibration, pinned_path

    path = pinned_path(project._beta.load_config().calibration, project.root)
    return path, load_calibration(path)


def _frame_ratio(data, calibration) -> float:
    """The plate's frame step over the file's ``frame_step_um`` (1 when impp uses the file's matrices as written)."""
    loaded = data.dataset_info.calibration.frame_step_um
    written = calibration.config.frame_step_um
    return 1.0 if loaded is None or written is None else float(loaded) / float(written)


def view_frame(project, selector: str, t: int, origin) -> frames.ViewFrame:
    """*selector*'s stored level-0 view as an :class:`impp_frames.ViewFrame`, checked against what impp recorded."""
    from impp._legacy.alignment_offsets import load_camera_alignment_4x4
    from impp.acquisition import CAMERA_BACKGROUND

    view = project.stored_view(selector, t=t, level=0)
    if view is None or view.geometry is None:
        raise SystemExit(f"{selector} at t={t} is not stored at level 0. Deskew it first (minutes and tens of GB):\n"
                         f"  pixi run python -m impp deskew {project.root} {selector} --t {t} --level 0")
    stream, data, _ = project._view_target(selector, t, 0, CAMERA_BACKGROUND)
    camera_id, camera_info, _ = data.camera(stream)
    calibration_loaded = data.dataset_info.calibration
    raw_shape = view.geometry["raw_shape"]
    pre = np.eye(4)
    for axis in calibration_loaded.camera_axes_to_flip_pre_deskew.get(camera_id, []):
        pre = frames.flip(raw_shape, (axis,)) @ pre
    after = np.eye(4)
    for axis in calibration_loaded.camera_axes_to_flip_post_deskew.get(camera_id, []):
        after = frames.flip(view.shape, (axis,)) @ after
    camera = load_camera_alignment_4x4(data.camera_alignment_config(), camera_id, stream.view)
    _, calibration = _pinned_calibration(project)
    frame = frames.ViewFrame(
        pre=pre, camera=np.eye(4) if camera is None else np.asarray(camera, float),
        deskew_loaded=np.asarray(data.dataset_info.deskew(stream.view), float),
        deskew_file=np.asarray(_file_matrix(calibration, stream.view), float),
        after=after, frame_ratio=_frame_ratio(data, calibration), top_shift=float(calibration.top_y_shift),
        view=stream.view, shape=tuple(view.shape), mirrored=bool(view.geometry.get("mirror_y")),
        origin=np.asarray(origin, float))
    frame.check(np.asarray(view.geometry["raw_to_view"], float))
    return frame


def _file_matrix(calibration, view: int) -> np.ndarray:
    from impp.calibration import _read_calibration_file

    return _read_calibration_file(calibration.path).deskew(view)


def _stream(project, selector: str):
    streams = project.acquisition.select(selector)
    if len(streams) != 1:
        raise SystemExit(f"{selector!r} names {len(streams)} streams; give one, such as 488nm@0")
    return streams[0]


def _camera_id(project, selector: str, t: int) -> int:
    from impp.acquisition import CAMERA_BACKGROUND

    stream, data, _ = project._view_target(selector, t, 0, CAMERA_BACKGROUND)
    return data.camera(stream)[0]


def origin_vox(project, a: str, b: str, t: int) -> np.ndarray:
    """B's origin in A's stored level-0 voxels, where the project's alignment puts it."""
    views = [project.stored_view(s, t=t, level=0) for s in (a, b)]
    return np.asarray(project._expected_offset(a, b, *views), float) / np.asarray(views[0].voxel_um, float)


def choose_mode(project, a: str, b: str, requested: str) -> str:
    sa, sb = _stream(project, a), _stream(project, b)
    if sa.position != sb.position:
        raise SystemExit("A and B are at different positions; pick on two views of one position")
    if sa.wavelength != sb.wavelength or sa.serial != sb.serial:
        if sa.view != sb.view:
            raise SystemExit("two cameras: give one view on each, such as 488nm@0 561nm@0")
        mode = "camera"
    elif sa.view == sb.view:
        raise SystemExit(f"{a} and {b} are the same view")
    else:
        mode = "shared-space" if (sa.view ^ 1) == sb.view else "top-bottom"
    if requested not in ("auto", mode) and not (mode == "shared-space" and requested == "slot"):
        raise SystemExit(f"{a} and {b} make a {mode} fit; --output-mode {requested} does not apply")
    return mode if requested == "auto" else requested


# --------------------------------------------------------------- solving

def _entry(view: int, matrix, frame: frames.ViewFrame, how: str) -> dict:
    return {view: {"matrix": np.asarray(matrix, float),
                   "solved_from": {"a_old": frame.deskew_file, "how": how,
                                   "post": frame.post(), "prev": frame.prev()}}}


def solve_views(mode: str, world, fixed: frames.ViewFrame, moving: frames.ViewFrame,
                partner: frames.ViewFrame | None) -> dict:
    """The ``view_transforms`` entries a world-space correction *world* (B's beads onto A's) makes."""
    if mode == "slot":
        return _entry(moving.view, frames.solve_slot(moving, world), moving, "slot: B takes the whole correction")
    if mode == "shared-space":
        half = frames.matrix_square_root(world)
        return {**_entry(fixed.view, frames.solve_slot(fixed, np.linalg.inv(half)), fixed,
                         "shared-space: A takes inv(sqrt(W))"),
                **_entry(moving.view, frames.solve_slot(moving, half), moving, "shared-space: B takes sqrt(W)")}
    return {**_entry(moving.view, frames.solve_slot(moving, world), moving, "top-bottom: B's objective moves"),
            **_entry(partner.view, frames.solve_slot(partner, world), partner, "top-bottom: B's objective moves")}


def solve_camera(args, world_a, world_b, moving: frames.ViewFrame, calibration_file: Path, camera_id: int) -> dict:
    fit = frames.camera_fit(moving, world_a, world_b)
    base = pick_projects.read_yaml(calibration_file)
    transforms = (base.get("camera_alignment") or {}).get("transforms") or {}
    key = frames.camera_key(transforms, camera_id, moving.view, frames.camera_3x3(moving.camera))
    views = args.camera_views or sorted({moving.view, moving.view ^ 1})
    print(f"[impp] camera fit: y {fit.rms()[0]:.2f}, x {fit.rms()[1]:.2f} raw px RMS; frame mismatch "
          f"{fit.frame_rms:.2f} frames RMS (pick error: a camera alignment cannot take it up)")
    entry = {"matrix": fit.matrix, "solved_from": {"a_old": frames.camera_3x3(moving.camera),
                                                   "how": "(Y, X) fit in B's raw frame",
                                                   "rms_raw_px": list(fit.rms()), "frame_rms": fit.frame_rms}}
    return {"camera_alignment": {key: {f"view_{v}": entry for v in views}}}


def on_apply(manager, context: dict) -> None:
    widget = manager.point_picker_widget
    world = widget._estimate_affine_transform()       # B's world (A's stored voxels) to A's
    if world is None:
        print("[impp] not enough pairs to fit")
        return
    if context["mode"] == "camera":
        used = [p for p in widget._complete_pairs().values() if p.enabled]
        world_a = np.array([p.layer1_coords for p in used], float)
        world_b = np.array([p.layer2_coords for p in used], float)
        entries = solve_camera(context["args"], world_a, world_b, context["moving"], context["calibration"],
                               context["camera_id"])
    else:
        entries = {"view_transforms": solve_views(context["mode"], world, context["fixed"], context["moving"],
                                                  context.get("partner"))}
    for kind, key, entry in pick_projects.iter_entries(entries):
        print(f"[impp] {pick_projects.describe_key(kind, key)}:")
        print(np.array2string(np.asarray(entry["matrix"]), precision=10, suppress_small=True))
    session = context["session"]
    if session is None:
        return
    result = {"calibration_entries": entries, "fit": _fit_summary(widget),
              "estimated_affine": {"matrix": world, "direction": "moving-to-fixed",
                                   "frame": "world: A's stored level-0 voxels, B drawn at its alignment offset"},
              "picks_file": widget.autosave_path.name}
    result["calibration_entries"] = pick_projects.snapshot_base_values(entries, context["calibration"])
    result["base_calibration"] = {"path": str(context["calibration"]),
                                  "sha256": pick_projects.file_sha256(context["calibration"])}
    print(f"[PickProject] Wrote {session.write_result(result)}")


def _fit_summary(widget) -> dict:
    report = widget.refresh_fit()
    if not report.ok:
        return {"pairs_used": report.n_used, "error": report.error}
    return {"units": "level-0 voxels", "pairs_used": report.n_used, "pairs_complete": report.n_complete,
            "rms": report.rms, "max_residual": report.residual_norms[report.max_pair], "max_pair": report.max_pair,
            "loo_rms": report.loo_rms, "worst_loo_pair": report.worst_loo_pair, "residuals": report.residual_norms,
            "leave_one_out": report.out_of_sample, "stability": report.stability, "warnings": report.warnings}


def estimator(mode: str):
    """View fits keep z-dependent rotations and z scale fixed, so x-y planes stay planes; a camera's diagnostics take a
    full affine (its matrix is solved in raw, :func:`impp_frames.camera_fit`)."""
    if mode == "camera":
        return estimate_affine_from_points, "full affine (diagnostics; the camera matrix is solved in raw)"
    return (partial(estimate_affine_from_points_components, fix_rotation=(False, True, True),
                    fix_scale=(True, False, False)), "view estimator (z rotations and z scale fixed)")


# ---------------------------------------------------------------- window

def _limits(view, down: int) -> tuple[float, float]:
    coarse = np.asarray(view.volume(view.downs))
    lo, hi = np.percentile(coarse[np.isfinite(coarse)], (1, 99.9))
    return float(lo), float(max(hi, lo + 1))


def add_view(viewer, project, selector: str, t: int, down: int, origin, colour: str, name: str) -> None:
    view = project.stored_view(selector, t=t, level=0)
    down = min(down, view.downs)
    kwargs = downsampled_layer_kwargs(2 ** down, full_res_translate=origin)
    viewer.add_image(view.volume(down), name=name, colormap=colour, blending="additive",
                     contrast_limits=_limits(view, down), scale=kwargs["scale"], translate=kwargs["translate"],
                     metadata={**kwargs["metadata"], "impp_project": str(project.root), "selector": selector,
                               "timepoint": t, "store": str(view.path)})


def preview_loader(project, t: int, sources: dict):
    """3D-preview provider: the finest stored halving of each view whose longest axis fits, placed in level-0 voxels
    like the displayed layers. *sources* maps a layer name to (selector, origin)."""

    def load(layer_name: str, max_size: int):
        if layer_name not in sources:
            return None
        selector, origin = sources[layer_name]
        view = project.stored_view(selector, t=t, level=0)
        down = next((d for d in range(view.downs + 1) if max(view.volume(d).shape) <= max_size), view.downs)
        kwargs = downsampled_layer_kwargs(2 ** down, full_res_translate=origin)
        return np.asarray(view.volume(down)), kwargs["scale"], kwargs["translate"]

    return load


def start_session(args, project, mode: str, label: str, calibration: Path, est_label: str):
    if args.no_pick_project:
        return None
    store = pick_projects.StoreProject.for_store(project.acquisition.path, args.picks_root)
    info = {"command": [Path(sys.argv[0]).name, *sys.argv[1:]], "impp_project": str(project.root),
            "fixed": args.a, "moving": args.b, "timepoint": args.t, "output_mode": mode, "estimator": est_label,
            "base_calibration": {"path": str(calibration), "sha256": pick_projects.file_sha256(calibration)},
            "coordinates": "A's stored level-0 voxels"}
    session = store.new_session(label, info)
    print(f"[PickProject] Session folder: {session.dir}")
    return session


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", type=Path, help="an impp project on the bead acquisition")
    parser.add_argument("a", help="the fixed stream, such as 488nm@0")
    parser.add_argument("b", help="the moving stream, such as 488nm@1")
    parser.add_argument("--t", type=int, default=0, help="the timepoint (default 0)")
    parser.add_argument("--down", type=int, default=0,
                        help="show the views at this 2x halving (picks stay in level-0 voxels)")
    parser.add_argument("--output-mode", choices=MODES, default="auto")
    parser.add_argument("--camera-views", type=int, nargs="+",
                        help="views a camera fit is written to (default: both of B's objective)")
    parser.add_argument("--picks", type=Path, help="a picks JSON to load into the table")
    parser.add_argument("--picks-root", type=Path, default=pick_projects.PICKS_ROOT)
    parser.add_argument("--no-pick-project", action="store_true", help="record no session")
    parser.add_argument("--display-filter", choices=("beads", "none"), default="beads",
                        help="'beads' (default) opens with Enhance Beads on: background, noise floor and isolated voxels "
                             "removed, then auto-contrast at gamma 0.5; 'none' opens on the stored values. Either can be "
                             "switched in the Point Picker tab; picks and fits are the same")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    impp = _impp()
    import napari

    project = impp.Project.open(args.project)
    mode = choose_mode(project, args.a, args.b, args.output_mode)
    calibration, _ = _pinned_calibration(project)
    origin = origin_vox(project, args.a, args.b, args.t)
    fixed = view_frame(project, args.a, args.t, np.zeros(3))
    moving = view_frame(project, args.b, args.t, origin)
    context = {"args": args, "mode": mode, "fixed": fixed, "moving": moving, "calibration": calibration}
    if mode == "top-bottom":
        partner = args.b.replace(f"@{moving.view}", f"@{moving.view ^ 1}")
        context["partner"] = view_frame(project, partner, args.t, origin_vox(project, args.a, partner, args.t))
    if mode == "camera":
        context["camera_id"] = _camera_id(project, args.b, args.t)
    est, est_label = estimator(mode)
    label = f"{mode}_{args.a}-{args.b}_t{args.t}"
    context["session"] = start_session(args, project, mode, label, calibration, est_label)
    print(f"[impp] {mode} fit of {args.b} onto {args.a}, t={args.t}; base calibration {calibration}")
    print(f"[impp] B is drawn at {np.round(origin, 2)} level-0 voxels from A, where the project's alignment puts it")

    viewer = napari.Viewer(title=f"bead picks: {args.b} onto {args.a} ({mode})")
    name_a, name_b = f"A {args.a} (fixed)", f"B {args.b} (moving)"
    add_view(viewer, project, args.a, args.t, args.down, np.zeros(3), "green", name_a)
    add_view(viewer, project, args.b, args.t, args.down, origin, "magenta", name_b)
    viewer.dims.point = tuple((np.asarray(fixed.shape, float) - 1) / 2)
    kwargs = {"autosave_dir": context["session"].dir} if context["session"] is not None else {}
    preview = preview_loader(project, args.t, {name_a: (args.a, np.zeros(3)), name_b: (args.b, origin)})
    manager = show_point_picker(viewer, layer1_name=name_a, layer2_name=name_b, affine_estimator=est,
                                preview_data_provider=preview, **kwargs)
    widget = manager.point_picker_widget
    if context["session"] is not None:
        widget.autosave_path = context["session"].picks_path
    if args.picks is not None:
        widget.load_points(str(args.picks))
    if args.display_filter == "beads":
        widget.enhance_beads()  # the button's own path, so Undo Enhance restores the stored values
    widget.affine_applied.connect(lambda _affine: on_apply(manager, context))
    napari.run()
    if context["session"] is not None and context["session"].remove_if_empty():
        print(f"[PickProject] No picks were made; removed {context['session'].dir}")


if __name__ == "__main__":
    main()
