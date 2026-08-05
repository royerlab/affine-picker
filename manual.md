# Creating a new transforms YAML for a DaXi microscope

This describes how to produce a new per-microscope transforms config
(e.g. `impp/impp/new_impp/config/transforms/daxi-a_<DATE>.yaml`) using
Ilan's point-picker alignment tool
(`load_zarr_views_for_registration.py`). The output file is consumed by
`impp`; per the DaXi pipeline it is the source of truth for
`camera_alignment` during `daxi2-deskew-fuse` (selected by
`--dataset-date`). This assumes the DaXi processing manual as background.

## What the file contains

One transforms YAML holds every alignment matrix for one microscope over
a date range:

- **`view_transforms`** — one 4x4 matrix per view (`0`-`3`), registering
  the four mirror configurations to a shared space. (View convention:
  0 = bottom-left, 1 = bottom-right, 2 = top-left, 3 = top-right;
  schematic, not literal.)
- **`camera_alignment.transforms`** — one 3x3 matrix per `(camera, view)`
  registering each non-reference camera onto the reference camera.
- Header metadata: `microscope`, `date_start`/`date_end`,
  `camera_name_to_id` (serial -> logical id), and per-camera
  `camera_axes_to_flip_pre_deskew`/`..._post_deskew`.

## Overview

You compute each matrix with a separate run of the tool, then assemble
the results.

For a 2-camera microscope like DaXi A:

- **One camera-alignment run per non-reference camera** (camera 0 is the
  reference, so one run for camera 1). A 4-camera scope needs one per
  non-reference camera.
- **One view-pair run per view pair** to produce the `view_transforms`
  matrices. Each run emits two views at once (e.g. views 2 and 3
  together).

Then paste all matrices into one file and install it.

> **Which output block to use differs by mode. This is a footgun.**
> The tool prints several representations of each result. For
> **view-pair** runs, use the `[TransformYAML] Copy-paste block:` (it is
> correct and pastes directly). For **camera** runs, do NOT use the
> copy-paste block or the "Equivalent embedded 4x4 (A_new)"; use the
> **world-space affine** exactly as printed (see Step 1).

## Step 1 — Camera alignment

> **If the dataset contains beads, skip the GUI.**
> `bead_camera_alignment.py` fits the same model from bead centroids and
> prints the whole `camera_alignment` block, all views in one run, with
> the sign already right. See
> [Step 1 without the GUI](#step-1-without-the-gui-bead-datasets) below.
> Pick points by hand when the sample has no beads, or to check a fit.

Registers a moving camera (one serial/wavelength) onto the fixed
reference camera for the same view. Uses separate fixed/moving zarr
paths and `--alignment-mode camera` (the default). Example (view 0,
camera 0 fixed -> camera 1 moving):

```
uv run python3 load_zarr_views_for_registration.py \
  --fixed-zarr-path  .../SN304410/488nm/pos0/0 \
  --moving-zarr-path .../SN302483/561nm/pos0/0 \
  --fixed-channel 0 --moving-channel 0 \
  --moving-camera-id 1 \
  --flip-fixed-x --no-flip-moving-y
```

The channel index selects the view. Read the mapping off the position
group's `.zattrs`: `omero.channels[i].label` reads like
`"Bottom Objective, Side A | v1_c1"`, and `v1`-`v4` are views 0-3 in the
order bottom-A, bottom-B, top-A, top-B.

The flip flags must reproduce the microscope's
`camera_axes_to_flip_pre_deskew`, because that is the frame the matrix
gets applied in. Axis 1 is Y and axis 2 is X, so a config carrying
`0: [2]` means the camera with logical id 0 needs `--flip-fixed-x` when
it is the fixed camera. Getting a flag wrong does not produce a subtly
worse fit, it produces one that will not converge below several pixels.

Pick corresponding points across the two layers in the GUI. The tool
prints a **world-space affine** (identical to the data-space affine when
the flips are applied to the data, which is what the tool does) as a
4x4:

```
[[ 1.            0.            0.            0.          ]
 [ 0.            0.9942680048 -0.0104417158 11.3612280404]
 [ 0.            0.009476072   0.9945267206  3.1706529672]
 [ 0.            0.            0.            1.          ]]
```

Take the **bottom-right 3x3** of that matrix and paste it unchanged.
That is the camera-alignment matrix:

```yaml
camera_alignment:
  transforms:
    camera_1:                     # the moving camera's id
      view_0:
        - [0.9942680048, -0.0104417158, 11.3612280404]
        - [0.0094760720,  0.9945267206,  3.1706529672]
        - [0.0000000000,  0.0000000000,  1.0000000000]
```

The reference camera gets identity matrices for all views.

> **Do not negate the translations.** The matrix is consumed as a forward
> map, from moving-camera voxel to reference-camera voxel, which is the
> same direction the point picker estimates. `canjica`'s
> `resample_affine` inverts whatever it is handed before resampling, so
> handing it an inverted matrix moves the data the wrong way and roughly
> doubles the mismatch instead of removing it.
> Negating only the translation while leaving the linear part alone is an
> approximation of that inverse, so on data it undoes nothing: measured
> on beads through `resample_affine` itself, the forward matrix takes the
> median bead mismatch from 6.4 px to 2.0 px, while the negated form
> leaves it at 6.4 px.

> **Do not use** the `[TransformYAML] Copy-paste block:` or the
> "Equivalent embedded 4x4 (A_new)" that the tool also prints for camera
> mode. Both convert the estimate back into unflipped data coordinates,
> and the pipeline needs it in flipped coordinates. The giveaway is a
> translation term close to the array's extent along the flipped axis
> (for example 2048 for a 2048-wide frame) together with a sign-flipped
> row, which is the flip folded into the matrix.

**Picking points well.** Spread the pairs across the full frame, not
just the region where features are densest. The scale and the
translation trade off against each other, so points confined to a band
pin down their combination only inside that band and extrapolate badly
outside it. A run with 7 pairs spanning Y 356-861 of a 1300-row frame
reached 1.1 px RMS on its own points, yet disagreed with a
whole-field fit by 7.9 px at the far corner. Aim for 15 to 20 pairs
reaching the top and bottom edges. Judge a run by its RMS (under about
2 px at full resolution is good) but read it together with the coverage,
since RMS only reports agreement on the points you picked.

## Step 1 without the GUI (bead datasets)

`bead_camera_alignment.py` replaces the point-picking above when the
sample contains beads. It detects beads in both cameras, matches them,
fits the in-plane (Y, X) affine with Z left untouched, and prints a
`camera_alignment` block covering every view, ready to paste. Full
resolution, four views, about two minutes:

```
ZARR=/hpc/instruments/royer.daxi1/JiaLe/alignment/\
20260405_488nm_561nm_10fps_50mW_beforealignment.ome.zarr

uv run python3 bead_camera_alignment.py \
  --fixed-zarr-path  $ZARR/SN304410/488nm/pos0 \
  --moving-zarr-path $ZARR/SN302483/561nm/pos0 \
  --moving-camera-id 1 \
  --flip-fixed-x \
  --channels 0,1,2,3
```

The arguments carry the same meaning as in the GUI run: point it at the
two position groups, name the moving camera's logical id, and set the
flip flags to match `camera_axes_to_flip_pre_deskew`. It prints per view:

```
channel 0 -> view 0
  seed phase-correlation (dY, dX)=[-222.  992.]: 64 pairs
  seed zero (dY, dX)=[0. 0.]: 1494 pairs
  seed zero wins with 1494 pairs
  outlier cut  8.00 px:  1454 pairs, RMS=3.019 max=8.226
  ...
  re-match with fitted model: 1676 pairs
  view 0: RMS=1.263 px  max=2.168 px  n=863
```

Read the RMS and the pair count. Both cameras see the same beads, so a
view should yield hundreds of pairs and settle near 1 to 1.5 px at full
resolution. A handful of pairs, or an RMS that stays above a few pixels,
almost always means a wrong flip flag or a channel with no beads in it
rather than a badly aligned camera. Two coarse starting guesses are
tried because a dense bead field can hand phase correlation a wrong
peak, visible above as the seed that found 64 pairs against 1494.

`--json-out` writes the matrices and residuals for scripting. Passing a
`--level` other than `0` fits on a downsampled pyramid level, which is
useful for a quick check: the linear part is scale-free, but the
translations come out in that level's pixels and need scaling before use.

The example above produced
`impp/impp/new_impp/config/transforms/daxi-a_2026-03-27.yaml`, whose
header records the command so the numbers can be regenerated.

## Step 2 — View-pair alignment

Produces the `view_transforms` matrices. Unlike Step 1 this uses a
single `--zarr-path` into a `deskewed.zarr`, `--alignment-mode view`, and
`--output-mode shared-space`. Annotated template (view 2 <-> view 3):

```
uv run python3 load_zarr_views_for_registration.py \
  --zarr-path /PATH/TO/PROJECT_FOLDER/deskewed.zarr/0 \
  --fixed-channel 2 \       # channel index IN deskewed.zarr (encodes view+wavelength)
  --moving-channel 3 \
  --fixed-camera-id 1 \     # must match the camera owning that channel (561nm -> 1)
  --moving-camera-id 1 \    # same value again
  --fixed-view-id 2 \       # logical view; NOT necessarily equal to the channel number
  --moving-view-id 3 \      #   (view 2 @ 488nm would be channel 6, but view id stays 2)
  --alignment-mode view \
  --no-flip-moving-y \
  --output-mode shared-space \
  --project-folder /PATH/TO/PROJECT_FOLDER   # same path as --zarr-path prefix
```

Here the `[TransformYAML] Copy-paste block:` IS the correct artifact.
Copy it directly — no sign surgery. It covers both views, plus a
companion `view_pair_offsets` block for `views_alignment.yaml`:

```yaml
view_transforms:
  2:
    - [1.0000000000, 0.2616579286, 0.0035239264, -248.1752520521]
    - [0.0000000000, 1.0011502788, 0.0001218013, 0.8326819207]
    - [0.0000000000, -0.0041436789, 1.0005969872, 59.0475569828]
    - [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]
  3:
    - [1.0000000000, -0.2611647404, -0.0023366980, 253.0572901506]
    - [0.0000000000, 0.9988551371, -0.0001203052, -1.0981308786]
    - [0.0000000000, 0.0041361850, 0.9994012024, -58.7500308004]
    - [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]
```

Repeat for each view pair to fill in all four `view_transforms` entries.

## Step 3 — Assemble the YAML

Paste the matrices from Steps 1-2 into one file and fill in the header:

```yaml
microscope: daxi-a
date_start: "2026-05-17"
date_end: "2029-06-03"

camera_name_to_id:              # verify direction: has a "this is inverted" TODO in the example
  "SN304410": 0
  "SN302483": 1

camera_axes_to_flip_pre_deskew:
  0: [2]
camera_axes_to_flip_post_deskew:
  0: []
  1: []

transform_type: per_view

view_transforms:                # from Step 2, all four views
  0: [ ... ]
  1: [ ... ]
  2: [ ... ]
  3: [ ... ]

camera_alignment:
  reference_camera: 0
  transforms:
    camera_0:                   # reference camera = identity for all views
      view_0:
        - [1.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0]
        - [0.0, 0.0, 1.0]
      view_1: [ ...identity... ]
      view_2: [ ...identity... ]
      view_3: [ ...identity... ]
    camera_1:                   # from Step 1 (world-space 3x3, pasted as printed)
      view_0: &camera1_v0
        - [ ... ]
      view_1: *camera1_v0       # reuse via anchor if a view shares a matrix
      view_2: &camera1_v2
        - [ ... ]
      view_3: *camera1_v2
```

Sharing one matrix across a view pair via a YAML anchor is a convenience,
not a constraint. The four views do differ: fitted independently on one
DaXi A bead stack their matrices displaced the field centre by amounts
spanning about 3 px, so a per-view matrix is worth having if you have the
correspondences to fit one.

Dates must not overlap. `find_transform_config` returns the first file
whose range contains the acquisition date, in filesystem glob order, so
two configs covering the same day makes the choice arbitrary. When adding
a config that supersedes an earlier one, set the earlier file's
`date_end` to the day before the new `date_start`.

## Step 4 — Install and use

Place the file at
`impp/impp/new_impp/config/transforms/daxi-<microscope>_<DATE>.yaml` and
select it with `--dataset-date`. Because `daxi2-deskew-fuse` sources
`camera_alignment` from this config (not `$WORKDIR/views_alignment.yaml`),
inter-camera registration is tuned here. Validate with a
`daxi2-deskew-fuse --slicing ::10` run and check camera/view alignment in
napari before a full batch.

## How this process could be simplified

- Make the camera-mode `[TransformYAML] Copy-paste block:` actually
  correct (the world-space 3x3, in flipped coordinates) so both modes can
  be copy-pasted the same way, and remove the misleading "Equivalent
  embedded 4x4 (A_new)" whose offset carries the frame extent, or fix
  whatever produces it.
- Have the tool write/append the YAML directly instead of manual
  paste-and-assemble across runs; ideally one driver that runs all
  required registrations for a microscope and emits the complete file.
  `bead_camera_alignment.py` does this for the camera-alignment half on
  bead data, so what remains is the view-pair half and writing the file
  rather than printing it.
- Remove redundant view-mode arguments: `--fixed-camera-id` /
  `--moving-camera-id` must be identical, and `--project-folder`
  duplicates the `--zarr-path` prefix.
- Fix and lock the `camera_name_to_id` direction (drop the "inverted"
  TODO) and auto-populate it from acquisition metadata.
- Auto-populate flip fields/flags from microscope geometry.
- Validate the assembled file against the `impp` schema before use.

## Settled questions

- **Camera translation sign.** Do not negate. `camera_alignment` is
  applied as `resample_affine(transform @ camera_align)`, and
  `resample_affine` inverts its argument internally, so the argument is a
  forward map. The point picker already estimates that direction. The
  earlier negated form was a hand-rolled inverse that dropped the linear
  part, and it leaves the cameras as misaligned as no correction at all.
- **World-space vs data-space affine.** Identical whenever the flips are
  applied to the array rather than to a display transform, which is what
  the tool does. Both are in flipped coordinates, which is the frame the
  pipeline wants.
- **Camera-alignment coverage.** One run per view per non-reference
  camera is worth doing. Reusing a matrix across a view pair costs about
  1 to 3 px of field-centre displacement on DaXi A.

## Open questions for Ilan

- **Point-picking targets.** Coverage and RMS guidance is in Step 1, but
  the acceptance threshold there (about 2 px RMS at full resolution) came
  from bead data. Sparser samples may deserve a looser number.
- **Residual field distortion.** After a best-fit affine, beads still
  disagree by around 1 to 2 px, varying across the frame, which is what
  a chromatic difference across the field would look like. An affine
  cannot absorb it. Whether it is worth a warp field for the camera pair,
  as `--apply-warp` already does for view pairs, is undecided.
- **Tool location/environment:** canonical path to
  `load_zarr_views_for_registration.py` and how to invoke it.

