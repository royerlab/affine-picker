# Creating a new transforms YAML for a DaXi microscope

This describes how to produce a new per-microscope transforms config
(e.g. `impp/impp/new_impp/config/transforms/daxi-a_<DATE>.yaml`) using
Ilan's point-picker alignment tool
(`load_zarr_views_for_registration.py`, in the
`napari-orthogonal-views` repo root). The output file is consumed by
`impp`; per the DaXi pipeline it is the source of truth for
`view_transforms` and `camera_alignment` during `daxi2-deskew-fuse`
(selected by acquisition date). This assumes the DaXi processing manual
as background.

## What the file contains

One transforms YAML holds every alignment matrix for one microscope over
a date range:

- **`view_transforms`** — one 4x4 matrix per view (`0`-`3`), registering
  the four mirror configurations to a shared space. Views 0/1 form one
  objective pair and views 2/3 the other. **The pipeline hardcodes views
  2,3 as the "bottom" pair** (`Constants.OLD_FORMAT_BOTTOM_VIEWS` in
  `new_impp/data/dataset.py`): they get an extra Y-flip and the
  `top_bottom_offsets` translation when placed into the shared frame,
  regardless of what the "Top/Bottom Objective" channel names say. The
  point-picker tool mirrors this hardcode on purpose — do not "fix"
  either side alone.
- **`camera_alignment.transforms`** — one 3x3 (Y,X) matrix per
  `(camera, view)` registering each non-reference camera onto the
  reference camera, applied in raw space before deskew.
- Header metadata: `microscope`, `date_start`/`date_end`,
  `camera_name_to_id` (serial -> logical id), and per-camera
  `camera_axes_to_flip_pre_deskew`/`..._post_deskew`.

> **Camera ids are two different things — this is THE footgun.**
> `camera_name_to_id` maps serials to the logical ids used for
> processing params (flip axes). But `camera_alignment` (both
> `reference_camera` and the `camera_<N>` keys) is keyed by the
> **dataset enumeration id** — the order cameras appear in the raw
> zarr. For the 2026-07-30 beads acquisition, SN302483 enumerates as
> dataset camera **0** even though `camera_name_to_id` maps it to 1.
> Two consequences:
> 1. Put each correction block under the **dataset id** of that camera.
> 2. `_normalize_camera_alignment` **forces the reference camera to
>    identity** — any corrections placed under the reference camera's
>    key are silently wiped.
> Always verify by opening the `views_alignment.yaml` the run emits in
> the output folder: it shows the normalized result actually used.

> **Date-range selection:** `find_transform_config` returns the *first*
> file whose range contains the acquisition date, in glob order. When
> adding a new file whose range overlaps an older one, narrow the older
> file's `date_end` — otherwise the old file may keep winning.

## Overview

Each matrix comes from a separate run of the tool; assemble the results
into one file. For a 2-camera microscope like DaXi A:

1. **Camera alignment** — one run per (non-reference camera, view pair).
2. **View-pair alignment** — one `shared-space` run per pair
   (0<->1, 2<->3); each emits both views' matrices.
3. **Cross-pair (top<->bottom) alignment** — one `top-bottom-*` run
   registering the 2/3 pair onto the 0/1 pair. Without this step the
   two pairs are each internally aligned but land far apart (hundreds
   of px); historically this was patched with hand-tuned
   `top_bottom_offsets` nudges, which a translation can never fully fix.
4. Assemble, install, validate with a zero-offset run.

> **Which output block to use differs by mode.** For **view-pair** and
> **top-bottom** runs, use the `[TransformYAML] Copy-paste block:`
> directly. For **camera** runs, use the **inverse of the world-space
> affine** (see Step 1) — not the copy-paste block.

## Step 1 — Camera alignment

Registers a moving camera (one serial/wavelength) onto the fixed
reference camera for the same view. Uses separate fixed/moving zarr
paths and `--alignment-mode camera` (the default). Example (view 0):

```
uv run python3 load_zarr_views_for_registration.py \
  --fixed-zarr-path  .../SN304410/488nm/pos0/0 \
  --moving-zarr-path .../SN302483/561nm/pos0/0 \
  --fixed-channel 0 --moving-channel 0 \
  --flip-fixed-x --no-flip-moving-y
```

Pick corresponding points across the two layers in the GUI. The tool
prints a **world-space affine** (moving -> fixed) as a 4x4:

```
[[ 1.            0.            0.            0.          ]
 [ 0.            0.9942680048 -0.0104417158 11.3612280404]
 [ 0.            0.009476072   0.9945267206  3.1706529672]
 [ 0.            0.            0.            1.          ]]
```

The config needs the **inverse** of this map. Take the bottom-right 3x3
and invert it (numpy: `np.linalg.inv(M3)`):

```yaml
camera_alignment:
  transforms:
    camera_1:                 # DATASET id of the moving camera (see footgun above)
      view_0:
        - [1.0056644090, 0.0105586524, -11.4590605055]
        - [-0.0095821944, 1.0054027959, -3.0789178625]
        - [0.0000000000, 0.0000000000,  1.0000000000]
```

The reference camera gets identity matrices for all views (anything
else you put there is wiped by normalization anyway).

> **The "negate the translations" folklore, explained.** The committed
> matrices are the *full inverse* of the printed estimate: reciprocal
> diagonal, sign-flipped off-diagonals, and translation `-L^-1 t`.
> Because the linear part is within ~1% of identity, "keep the 3x3,
> negate the last column" approximates the inverse to ~0.1 px — that is
> why the old recipe worked. Do the proper inversion; it is exact and
> removes the mystery. (Equivalently, run with `--output-inverse`.)
>
> Still **do not use** the camera-mode `[TransformYAML] Copy-paste
> block:` or the "Equivalent embedded 4x4 (A_new)": the ~2043
> translation is a display-flip (X extent) folded into the offset by
> the flipped-layer conversion — a different frame than the config
> expects.

Repeat per view pair: in practice one 2D matrix serves views 0/1 and
another serves views 2/3 (reuse via YAML anchors), so a 2-camera DaXi
needs two camera runs (e.g. view 0 and view 2).

## Step 2 — View-pair alignment

Produces the within-pair `view_transforms` matrices. Uses a single
`--zarr-path` into a `deskewed.zarr`, `--alignment-mode view`, and
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

Here the `[TransformYAML] Copy-paste block:` IS the correct artifact —
copy it directly, no sign surgery. It covers both views, plus a
companion `view_pair_offsets` block (set those offsets to zero in
`views_alignment.yaml`, as the block instructs). Repeat for the other
pair.

The solve conjugates the picked correction through the pipeline's POST
factors (`A_new = inv(POST) @ correction @ POST @ A_old`), so it is only
valid if the `--project-folder` config matches what actually generated
the loaded `deskewed.zarr`. For the bottom pair, the pipeline applies the
view-3 pair offset before its later top/bottom Y-flip; the solver converts
that offset into output coordinates before splitting it across views 2/3.

## Step 3 — Cross-pair (top<->bottom) alignment

After Step 2 each pair is internally consistent but the 2/3 pair is not
yet registered onto the 0/1 pair. Pick points across the pairs (fixed =
a 0/1 view, moving = a 2/3 view) and use one of two modes:

**`top-bottom-direct`** — the easy case: the loaded `deskewed.zarr` was
generated with the *same* `view_transforms` you are updating (i.e. you
regenerated after Step 2). GUI run, same layout as Step 2 but:

```
  --fixed-channel 0 --fixed-view-id 0 \
  --moving-channel 2 --moving-view-id 2 \
  --output-mode top-bottom-direct
```

The copy-paste block emits updated matrices for **both** moving-pair
views (2 and 3) — the same output-space correction applied to each
through its own POST (which includes the hardcoded bottom-view Y-flip
and any `top_bottom_offsets`), so the pair's internal alignment is
preserved. Existing `views_alignment.yaml` offsets stay as they are.

**`top-bottom-rebased`** — the problematic case: views 2/3 already have
*new* pending transforms (e.g. a fresh Step 2 result) but your
top<->bottom point pairs were picked on a dataset generated with the
*old* transforms. Instead of regenerating and re-picking, the tool
rebases the saved points through the old->new frame change and then
solves as above. Headless (no GUI):

```
uv run python3 load_zarr_views_for_registration.py \
  --zarr-path /PATH/TO/OLD_PROJECT/deskewed.zarr/0 \
  --current-zarr-path /PATH/TO/CURRENT_PROJECT/deskewed.zarr/0 \
  --fixed-channel 0 --fixed-view-id 0 \
  --moving-channel 2 --moving-view-id 2 \
  --fixed-camera-id 1 --moving-camera-id 1 \
  --alignment-mode view \
  --output-mode top-bottom-rebased \
  --point-pairs-json saved_pairs_v0_v2.json \
  --project-folder /PATH/TO/PROJECT_WITH_NEW_TRANSFORMS \
  --transform-yaml /PATH/TO/IMMUTABLE_NEW_TRANSFORM_SNAPSHOT.yaml \
  --points-project-folder /PATH/TO/OLD_PROJECT \
  --save-rebased-points rebased_pairs.json
```

`--points-project-folder` describes the dataset the points were picked
on (old transforms + offsets); `--project-folder` provides the current
alignment metadata. Use `--transform-yaml` when the new transforms are in an
immutable snapshot rather than that folder's live `impp_settings.yaml`; this
avoids accidentally applying the same correction a second time after the live
settings have been overwritten. `--current-zarr-path` supplies the regenerated
final canvas shape used by the new pipeline transforms. The tool combines it
with `top_bottom_offsets` to recover the pre-placement deskewed view shape,
because views 2/3 are Y-flipped before the pipeline grows the final canvas.
If the current Zarr does not exist yet, pass
`--current-shape Z Y X`. When neither option is given, the tool tries
`job.output_path` from the current project's `impp_settings.yaml` and
stops if it cannot resolve a current shape. The optional
`--save-rebased-points` writes the pairs mapped into the new frame so
they can be reloaded in the GUI on the regenerated dataset for
fine-tuning.

Caveats for the rebased mode:

- The rebase is purely model-based (`M = FULL_new @ inv(FULL_old)` per
  view). It is exact for the transform change itself when the old and
  current final Zarr shapes and their matching `top_bottom_offsets` are
  supplied correctly. A different code checkout
  between runs can still introduce factors the model cannot see.
  Regenerate and verify; re-pick if residuals look wrong.
- Point pairs are only as good as their spread: use **>= 8-10 pairs
  covering the full field in all three axes**. Near-coplanar picks
  (e.g. all beads at similar z-vs-y) leave shear/rotation directions
  unconstrained and the estimator can return a wildly wrong matrix with
  deceptively small residuals. Check the printed per-axis residuals; a
  well-spread set fits at ~2-7 px RMS.

## Step 4 — Assemble the YAML

Paste the matrices from Steps 1-3 into one file and fill in the header:

```yaml
microscope: daxi-a
date_start: "2026-07-30"
date_end: "2029-06-03"

camera_name_to_id:              # serial -> logical id (processing params/flips ONLY;
  "SN304410": 0                 #   camera_alignment uses DATASET enumeration ids)
  "SN302483": 1

camera_axes_to_flip_pre_deskew:
  0: [2]
camera_axes_to_flip_post_deskew:
  0: []
  1: []

transform_type: per_view

view_transforms:                # Step 2 within-pair + Step 3 cross-pair folded in
  0: [ ... ]
  1: [ ... ]
  2: [ ... ]
  3: [ ... ]

camera_alignment:
  reference_camera: 0           # DATASET id of the reference camera
  transforms:
    camera_0:                   # reference camera: identity (forced anyway)
      view_0: &ident
        - [1.0, 0.0, 0.0]
        - [0.0, 1.0, 0.0]
        - [0.0, 0.0, 1.0]
      view_1: *ident
      view_2: *ident
      view_3: *ident
    camera_1:                   # DATASET id of the moving camera; inverted 3x3s from Step 1
      view_0: &cam1_v01
        - [ ... ]
      view_1: *cam1_v01
      view_2: &cam1_v23
        - [ ... ]
      view_3: *cam1_v23
```

## Step 5 — Install and validate

Place the file at
`impp/impp/new_impp/config/transforms/<microscope>_<DATE>.yaml` (narrow
the previous file's `date_end` if ranges overlap). Then run a
**zero-offset validation** so nothing masks the matrices:

```
CONDA_OVERRIDE_CUDA=12.0 pixi run impp daxi2-deskew-fuse RAW.ome.zarr \
  -p DEV --suffix mycheck --operation deskew-only \
  --microscope daxi-a --non-interactive --skip-vp --skip-tb
```

(`--skip-vp --skip-tb` with no pre-existing `views_alignment.yaml`
means all manual offsets are zero; `CONDA_OVERRIDE_CUDA` is needed on
non-GPU submit nodes.) Check in napari or with channel-overlay MIPs:

- within-pair (ch0 vs ch1, ch2 vs ch3): should be ~1-3 px;
- cross-pair (ch0 vs ch2): the Step 3 result — expect roughly the RMS
  of your picked pairs (~5-10 px with 9 decent picks);
- cross-camera (ch0 vs ch4): the Step 1 result — expect ~1-4 px.

Also open the run's emitted `views_alignment.yaml` and confirm the
camera_alignment block landed on the intended dataset camera ids.

**Migration caveat:** any existing production folder still carrying
hand-tuned `top_bottom_offsets`/`view_pair_offsets` nudges must have
them zeroed when switching to a config whose matrices already include
those corrections — otherwise they are applied twice.

## How this process could be simplified

- Make the camera-mode `[TransformYAML] Copy-paste block:` emit the
  inverted 3x3 in config form directly, and drop the misleading
  "Equivalent embedded 4x4" with the display-flip offset.
- Key `camera_alignment` by serial (or by `camera_name_to_id`) instead
  of dataset enumeration order, or at least warn when the two diverge.
- Make the pipeline's bottom-view set config-driven end to end (it
  already *parses* Top/Bottom from channel names, then ignores it in
  `get_bottom_views`); until then both pipeline and tool must keep the
  {2,3} hardcode in lockstep.
- Have the tool write/append the YAML directly instead of manual
  paste-and-assemble; ideally one driver that runs all registrations
  for a microscope and emits the complete file.
- Remove redundant view-mode arguments: `--fixed-camera-id` /
  `--moving-camera-id` must be identical, and `--project-folder`
  duplicates the `--zarr-path` prefix.
- Auto-populate flip fields/flags from microscope geometry.
- Validate the assembled file against the `impp` schema before use.

## Resolved questions (previously open)

- **Camera translation sign flip:** the config wants the inverse of the
  tool's moving->fixed estimate; negating the translations approximates
  the inverse because the linear part is near identity. Invert properly
  and the folklore disappears.
- **World-space vs data-space affine:** identical when no display flips
  are active. With flips, the `[TransformYAML]` path converts back to
  original data coordinates itself — but in camera mode that conversion
  is what produces the flip-contaminated block, hence "use the inverted
  world-space 3x3" for cameras.
- **Camera-alignment coverage:** one run per (non-reference camera,
  view pair); views within a pair share the matrix via anchors.
- **`camera_name_to_id` direction / "inverted" TODO:** the mapping
  itself only routes processing params; the confusion came from
  `camera_alignment` being keyed by dataset enumeration ids, which for
  some acquisitions are the opposite of the logical ids.
- **Point-picking guidance:** >= 8-10 pairs, spread across the full
  field in z, y AND x; avoid clustered/coplanar sets; expect ~2-7 px
  RMS from careful picks and treat the per-axis residuals as the
  acceptance signal. The final cross-pair alignment quality equals the
  picked-pair fit quality, so picks are the accuracy floor.
- **Tool location:** `load_zarr_views_for_registration.py` in the
  `napari-orthogonal-views` repo root; run with that repo's environment
  (`uv run python3 ...`).

## Still open

- Whether the halide `resample_affine` convention (forward vs pull) is
  the principled root of the camera-inverse rule, or whether the fixed /
  moving roles in the historical camera runs were simply swapped —
  empirically settled either way, but worth pinning down in the code.
- The ~1.4% y-scale residual observed between views 0 and 1 on the
  2026-07-30 beads data: within-pair solves should catch this — check
  whether Step 2 picks spanned enough y-range.
