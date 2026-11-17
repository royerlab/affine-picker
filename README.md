# napari-orthogonal-views

[![License BSD-3](https://img.shields.io/pypi/l/napari-orthogonal-views.svg?color=green)](https://github.com/AnniekStok/napari-orthogonal-views/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-orthogonal-views.svg?color=green)](https://pypi.org/project/napari-orthogonal-views)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-orthogonal-views.svg?color=green)](https://python.org)
[![tests](https://github.com/AnniekStok/napari-orthogonal-views/workflows/tests/badge.svg)](https://github.com/AnniekStok/napari-orthogonal-views/actions)
[![codecov](https://codecov.io/gh/AnniekStok/napari-orthogonal-views/branch/main/graph/badge.svg)](https://codecov.io/gh/AnniekStok/napari-orthogonal-views)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-orthogonal-views)](https://napari-hub.org/plugins/napari-orthogonal-views)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A napari plugin for orthogonal views with synced events, crosshairs, and **interactive point-based affine registration** between two image layers.

----------------------------------

![orthoviews](https://github.com/user-attachments/assets/9d1ea326-866d-4af7-9ea6-8e56046cf6f2)

This plugin builds on the upstream [napari-orthogonal-views](https://github.com/AnniekStok/napari-orthogonal-views) by adding a **point picker** for manually selecting matched landmarks across two image layers and computing an affine transform to align them. The orthogonal views, crosshairs, and syncing infrastructure come from the original plugin; the registration workflow is the main addition.

## Installation

```
uv add https://github.com/shinzlet/napari-orthogonal-views.git
```

For DaXi bead calibration with impp, run `pick_impp_views.py` in impp's environment: [TRANSFORMS_YAML_HOWTO.md](TRANSFORMS_YAML_HOWTO.md) walks through it.

## Point picker & affine registration

The quickest way to get started is `show_point_picker`, which opens orthogonal views with crosshairs, zoom sync, and center sync enabled, and switches to the Point Picker tab:

```python
import napari
from napari_orthogonal_views import show_point_picker

viewer = napari.Viewer()
viewer.add_image(fixed, name="Fixed", colormap="green", blending="additive")
viewer.add_image(moving, name="Moving", colormap="magenta", blending="additive")

manager = show_point_picker(viewer, layer1_name="Fixed", layer2_name="Moving")
```

### Translation offsets

If your two images are offset by a known coarse translation, set `translate` on the moving layer when adding it to the viewer. The point picker uses napari's `layer.translate` property to render the images at their true world-space positions without allocating a large zero-padded bounding box:

```python
offset = np.array([50, 30, 20])  # (z, y, x) offset of moving image
viewer.add_image(moving, name="Moving", colormap="magenta",
                 blending="additive", translate=offset)
```

Layer 1's `(0, 0, 0)` is treated as the global origin. When the affine is applied, the output is automatically sized to the bounding box of the transformed moving image, so no data is clipped.

### Custom affine estimator

By default, `show_point_picker` uses `estimate_affine_from_points`, which estimates a full affine (rotation, shear, scale, and translation). You can pass a custom estimator function with the signature `(source_points: ndarray, target_points: ndarray) -> ndarray` (returning a homogeneous affine matrix):

```python
from napari_orthogonal_views import (
    show_point_picker,
    estimate_affine_from_points_no_scale,
)

manager = show_point_picker(
    viewer, layer1_name="Fixed", layer2_name="Moving",
    affine_estimator=estimate_affine_from_points_no_scale,
)
```

Built-in estimators:

| Function | Description |
|----------|-------------|
| `estimate_affine_from_points` | Full affine with all degrees of freedom (default) |
| `estimate_affine_from_points_no_scale` | Constrains the diagonal to 1 — only shear, rotation, and translation |

### Picking pairs

The Point Picker tab lists every pair with its fixed and moving coordinates, and refits the affine after each edit, so you can judge the fit before resampling anything. Its top line says which colour is which image (for example **Fixed** is green, **Moving** is magenta), read from the layers' colormaps; the canvas overlay repeats it.

1. Navigate to a bead in the fixed image and press **T** to move the crosshair to the mouse. Refine it in the orthogonal views if needed.
2. Press **Q** to store the crosshair as the active pair's fixed point (a pair is created if there is none).
3. Find the same bead in the moving image, press **T**, then **W** to store the moving point.
4. Press **N** for the next pair and repeat. At least 4 pairs are needed; more, spread through the volume, is better.
5. Click **Apply Estimated Affine** to resample the moving image onto the fixed one. You can keep adding pairs and re-apply.
6. **Reset Transform** restores the original image. **Reset (keep view)** keeps the resampled image and clears the pairs, so you can pick a second round on top of a partial alignment.

The hotkeys are listed at the top of the tab and, by default, on the canvas together with the active pair's state. They work while the mouse is over the main view or an orthogonal view:

| Key | Action |
|-----|--------|
| T | Move the crosshair to the mouse |
| Q / W | Store the crosshair as the active pair's fixed / moving point |
| Shift+Q / Shift+W | Go to the active pair's fixed / moving point |
| N | New pair |
| [ / ] | Previous / next pair |
| U | Use or skip the active pair in the fit |
| Ctrl+Z | Undo the last add, set, delete, skip or load |
| Shift+drag | Slide the moving image in that view's two axes |
| C | Switch the view under the main view between xz and yz |

The **Show** and **Update** buttons in each row do the same as the go-to and store keys for that row.

Shift+drag works in the main view and the orthogonal view, moves the image in whole voxels of the view, and adds to the **Nudge Offset** below the table (**Reset Nudge** puts the image back). The nudge only moves the image on screen. A moving point is stored at its position in the moving image, so the pairs, the fit and saved picks are the same whatever the nudge. Use it to bring a bead pair close enough to compare by eye before picking.

### Judging the fit

Each row shows two errors, in the same units as the coordinates:

- **Resid** is the distance between the fixed point and the moving point mapped through the current fit.
- **LOO** (leave-one-out) is that distance under a fit made without the pair. A mismatched pair drags the full fit towards itself, so its own residual can look small; its LOO residual does not. For a skipped pair both columns show its distance under the full fit.

Rows above the **Flag** threshold are highlighted, and the table can be ordered by either error. Unchecking **Use** (or pressing **U**) leaves a suspect pair out of the fit without deleting it, so you can watch the RMS change.

Below the table, the **Fit** box summarises the fit and warns about problems:

- RMS and worst pair, for the fit and for leave-one-out.
- **Stability**: the picks are refitted with simulated pick noise of half a displayed voxel, and the box reports how far the corners of the moving image move. A large value means the picks do not constrain the fit, even when the residuals are small.
- How much of the moving image the picks span along each axis.
- The fit factored into scale, rotation, shear and translation, with a warning when it is far from identity (scale outside 0.8 to 1.25, rotation above 10 degrees, or shear above 0.2).
- A warning when the picks are nearly coplanar, and the reason when too few pairs are present or the estimator fails.

### Picking on a downsampled pyramid level

Large volumes are easier to navigate at a coarse pyramid level. Add those layers with `downsampled_layer_kwargs(factor)`, which sets the layer's scale to the downsampling factor and shifts it by half a coarse voxel. Napari world coordinates are then full-resolution voxels, so the picks, residuals, saved files and estimated affine are all in full-resolution voxels, whichever level is on screen:

```python
from napari_orthogonal_views import downsampled_layer_kwargs

level2 = downsampled_layer_kwargs(4)  # level 2 of a 2x pyramid
viewer.add_image(fixed_level2, name="Fixed", **level2)
viewer.add_image(moving_level2, name="Moving", **level2)
```

A coarse voxel `i` covers full-resolution voxels `4i` to `4i + 3`, so a pick on it is stored at its centre, `4i + 1.5`. `pick_impp_views.py` does this when given `--down`: it shows impp's stored halvings and keeps picks and results in level-0 voxels.

### Making beads easier to see

Bead views have a background of a few hundred counts, with a gradient along the light sheet, while the beads range from about 1,000 counts to saturation. On fixed 0 to 65,535 contrast limits most beads sit at a few percent of the range, and lowering gamma to find them lifts the background too. `enhance_beads` removes the background for display:

```python
from napari_orthogonal_views.bead_display import DISPLAY_GAMMA, enhance_beads

shown, limits = enhance_beads(volume, level_factor=4)  # level_factor: the pyramid level's downsampling
viewer.add_image(shown, contrast_limits=limits, gamma=DISPLAY_GAMMA, **downsampled_layer_kwargs(4))
```

It subtracts each 8 x 32 x 32 block's 10th percentile (in full-resolution voxels; beads are too sparse to move it), interpolated linearly between blocks, then a floor one noise standard deviation above the typical background, and clips at zero. About 85% of the background comes out black; most of the rest is isolated voxels, so any voxel with fewer than 8 of its 26 neighbours above zero is set to zero too (a bead is a connected blob and keeps its sharp apex, which a median filter would blur). It also returns contrast limits up to the 99.95th percentile of the result. Bead positions do not change, so picks and fits are the same as on the raw data. A full-resolution view (800 x 1300 x 2048) takes about 12 s.

At the top of the Point Picker tab, **Enhance Beads** applies it to both images, then auto-contrasts them (like the layer controls' **once** button, the range of the shown slice) and sets gamma to 0.5. **Undo Enhance** restores the raw data and the contrast and gamma you had. Both work at any point. While an affine is applied, the moving image's source (what **Apply** resamples from) is filtered or restored and the applied affine is resampled again, which takes as long as an **Apply**; **Reset Transform** and **Reset (keep view)** then keep the state you chose. The raw data stays in memory while the filter is on, so each image takes twice its memory.

`pick_impp_views.py` opens with Enhance Beads on; `--display-filter none` opens on the stored values.

### Saving, autosave and loading

**Save Points** writes a JSON file with every pair, including unfinished and skipped ones, plus what it was picked on: each layer's shape, scale and metadata (source path, channel, timepoint, flips), the coordinate frame and the estimator. Picks are also autosaved after every edit to `~/.napari-point-picker/autosave/<time>_<moving layer>.json` (the path is shown in the tab; pass `autosave_dir=` to change it, or `None` to turn it off), so a crash loses nothing. A second copy is written after every edit to `/tmp/point-picker-<user>/` on the machine running napari (pass `backup_dir=` to change it, or `None` to turn it off), so the picks survive even when the Save dialog or the autosave location fails.

**Load Points** reads these files and the older format (complete pairs only). It warns when the file does not match the open data, for example a different channel, flips or full-resolution shape, or picks saved in a different coordinate frame.

### 3D preview

The **3D preview** buttons open a separate napari window with both images rendered in 3D (maximum intensity projection, additive) and the picks drawn as labelled points:

- **Raw data** shows both images at their original positions.
- **With current estimate** maps the moving image through the current fit, applied or not, using napari's own affine (no resampling).

The preview is downsampled until its longest axis fits the size you set (384 voxels by default). By default the layers are strided. Pass `preview_data_provider=` to `show_point_picker` to supply a coarse pyramid level instead; `pick_impp_views.py` does this, choosing the finest stored halving that fits. Loading runs in the background, and the window opens when it is done.

### Pick projects and calibration output

`pick_impp_views.py` records every run as a session in `picks/<store>/` in this repository (see [picks/README.md](picks/README.md)). It holds small text files only, and is meant to be committed. The picks autosave into the session, and each **Apply** writes `result.yaml`, which contains:

- the fit: RMS, worst pair, per-pair and leave-one-out residuals, stability, warnings
- the calibration entries the fit changes (`view_transforms[2]`, `camera_alignment.transforms.camera_1.view_0`, ...), with the A_old each was solved against
- the base calibration file it belongs to (`--base-calibration`, or `--transform-yaml` when that is a calibration file), with its hash

In camera mode the entry is the inverse of the fitted (Y, X) 3x3, and it is recorded as an entry only when `--camera-alignment-key camera_N` names the dataset enumeration id of the camera it applies to (with `--camera-alignment-views` for the views it serves). The copy-paste block is still printed, but nothing needs to be read off the terminal. Accept the sessions you trust, then assemble a complete calibration file:

```
python pick_projects.py list
python pick_projects.py accept picks/<store>/sessions/<session>
python pick_projects.py assemble picks/<store>
# wrote picks/<store>/calibration/daxi-a_2026-08-07_picked.yaml
```

The assembled file is the base calibration with the accepted entries written in. impp reads it as is (`Project.create(..., calibration=path)`), or it can be filed into impp's calibration library. Assembly refuses an entry, unless given `--force`, when:

- the base calibration has changed since the session recorded it, or
- the entry was solved against an A_old that differs from the base's value for that entry. This happens when the store was deskewed with another calibration; use that calibration as the base.

Assembly also takes the beads' top/bottom spacing out of the result. A cross-objective fit puts it into view 2's y translation, and left there it would cut that many rows of the top view at deskew. The spacing is set per sample by the top/bottom offset anyway. Whole rows beyond 10 are removed from view 2, and from view 3 only when view 3 carries the same shift. `assembly.yaml` records the rows removed.

### Programmatic access

```python
# Retrieve the complete pairs used in the fit (dict keyed by layer name,
# in each layer's data space: full-resolution voxels when picking on a
# downsampled level)
pairs = manager.get_registration_points()
# {"Fixed": [(z,y,x), ...], "Moving": [(z,y,x), ...]}

# The live fit and its diagnostics (residuals, leave-one-out, stability)
report = manager.point_picker_widget.refresh_fit()
print(report.rms, report.out_of_sample)

# Save or load a pick file
manager.point_picker_widget.save_points("picks.json")
manager.point_picker_widget.load_points("picks.json")

# Get the estimated affine matrix (4x4 homogeneous for 3D, in data coordinates)
affine = manager.get_estimated_affine()

# Load previously saved point pairs (in data-space coordinates)
manager.load_registration_points(pairs)
```

See `demo_point_picker.py` for a basic example and `demo_point_picker_offset.py` for an example with a translation offset.

## Orthogonal views

Commands are available in Views > Commands Palette (Cmd+Shift+P):
  - Show Orthogonal Views
  - Hide Orthogonal Views
  - Toggle Orthogonal Views
  - Remove Orthogonal Views

Or from the console:

```python
from napari_orthogonal_views.ortho_view_manager import show_orthogonal_views
show_orthogonal_views(viewer)
```

The main (xy) view fills the top of the left column, and one orthogonal view sits under it: xz by default, or yz. Press **C** with the mouse over a view, or use the button in the Controls tab, to switch between them. The settings tabs (Controls, Screen Recorder, Point Picker) fill the right column. The panes can be resized by dragging the splitter handles. Screenshots and recordings can include either orthogonal view, shown or not. Pressing **T** centers all views on the mouse position. All events (including label painting) are synced across views via a shared data array.

### Syncing properties

By default all layer properties are synced. For finer control, call `set_sync_filters` *before* showing orthoviews:

```python
from napari_orthogonal_views.ortho_view_manager import _get_manager
from napari.layers import Tracks, Labels

m = _get_manager(viewer)
m.set_sync_filters({
    Tracks: {"forward_exclude": "*", "reverse_exclude": "*"},
    Labels: {"forward_exclude": "contour"},
})
```

### Screen recording

The **Screen Recorder** tab can save a stitched screenshot or sweep along an axis to produce an AVI video.

## Known issues

- Deprecation warnings on `Window._qt_window`, `LayerList._get_step_size`, `LayerList._get_extent_world` (suppressed for now).
- After removing the OrthoViewManager with `delete_and_cleanup` (Remove Orthogonal Views command), the canvas may become temporarily unresponsive. Clicking outside of napari and back usually fixes this.

## Contributing

Contributions are very welcome. Tests can be run with [tox]; please ensure coverage at least stays the same before submitting a pull request.

## License

Distributed under the terms of the [BSD-3] license, "napari-orthogonal-views" is free and open source software.

## Issues

If you encounter any problems, please [file an issue](https://github.com/AnniekStok/napari-orthogonal-views/issues/) along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
