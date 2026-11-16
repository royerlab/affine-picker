# Calibrating a DaXi microscope from beads with impp

A DaXi calibration file (impp's `impp/impp/transforms/daxi-<scope>_<YYYY-MM-DD>.yaml`) holds a deskew matrix per view
(`view_transforms`, raw voxel to deskewed voxel) and a (Y, X) camera alignment per camera and view
(`camera_alignment.transforms`). This page fits new matrices from a bead acquisition with `pick_impp_views.py`, which
reads views an impp project has deskewed and writes entries in the pinned file's own frame.

## What a calibration can and cannot hold

- **Camera alignment** registers each non-reference camera onto the reference camera, in raw (Y, X) before deskew. It
  holds for every acquisition in the date range.
- **Within an objective** (views 0 and 1, views 2 and 3) the deskew matrices carry the shear, scale and rotation
  between the pair. impp still measures the pair's offset per acquisition (`Project.align`), so a small leftover shift
  is harmless.
- **Across the objectives** (0 and 2) only the linear part carries over. The two objectives move independently, so
  the top/bottom offset is set per sample (by eye in impp's placement window). A cross-objective fit folds the beads'
  spacing into view 2's y translation; `pick_projects.py assemble` takes it out again, and impp does the same at load.

## Steps

Run everything from an impp checkout, in its environment (`pixi run`), on a node with a desktop session for the
windows. `PICK=external/napari-orthogonal-views`.

1. **Make a project on the beads**, pinned to the calibration you are correcting (by default the one in force on the
   beads' date):

   ```python
   import impp
   proj = impp.Project.create("/hpc/instruments/.../20261103_beads_02um_488nm_561nm.ome.zarr",
                              "/hpc/scratch/.../beads_20261103")
   ```

2. **Deskew the views you will pick on, at level 0** (minutes and 10 to 30 GB per view; say so before starting):

   ```
   pixi run python -m impp deskew PROJECT 488nm@0 488nm@1 488nm@2 488nm@3 561nm@0 561nm@2 --t 0 --level 0
   ```

3. **Pick, one run per fit.** A is fixed, B moves. B is drawn where the project's alignment puts it.

   | Fit | Command | Mode (chosen from the pair) |
   |---|---|---|
   | Camera, bottom objective | `pick_impp_views.py PROJECT 488nm@0 561nm@0` | `camera`, written to views 0 and 1 |
   | Camera, top objective | `pick_impp_views.py PROJECT 488nm@2 561nm@2` | `camera`, written to views 2 and 3 |
   | Within the bottom pair | `pick_impp_views.py PROJECT 488nm@0 488nm@1` | `shared-space`: half to each view (`--output-mode slot` moves B only) |
   | Within the top pair | `pick_impp_views.py PROJECT 488nm@2 488nm@3` | `shared-space` |
   | Top onto bottom | `pick_impp_views.py PROJECT 488nm@0 488nm@2` | `top-bottom`: views 2 and 3 move together |

   Prefix each with `pixi run python $PICK/`. Pick 8 to 10 or more beads spread over the whole field in z, y and x
   (near-coplanar picks leave shear and rotation unconstrained), watch the live residuals, and press **Apply**. Each
   Apply writes the session's `result.yaml` and prints the entries. The camera reference camera's alignment is always
   identity: make the reference camera A.

   Fit cameras first. Picks measure against the matrices the project deskewed with, so a view fit made on a project
   whose camera alignment then changes still holds (camera alignment acts before deskew, view matrices after), but a
   second fit of the same view must be made on a project pinned to the first fit's file.

4. **Accept and assemble**:

   ```
   pixi run python $PICK/pick_projects.py list
   pixi run python $PICK/pick_projects.py accept $PICK/picks/<store>/sessions/<session>
   pixi run python $PICK/pick_projects.py assemble $PICK/picks/<store>
   ```

   Assembly refuses an entry solved against a matrix that differs from the base file's (the session was picked on a
   project pinned to another calibration).

5. **Check it.** Pin the assembled file in a new project on the beads and on a sample, and look at single planes:

   ```python
   check = impp.Project.create(RAW, DIR, calibration=ASSEMBLED, reason="bead fit 2026-11-03, checking")
   ```

   Within a pair, beads should coincide to 1 to 3 voxels; across cameras to 1 to 4. File the matrices into impp's
   library (`impp/impp/transforms/`, narrowing the previous file's `date_end`) once a fusion pinned to them looks right.

## How the solve works

impp stores a view so that raw voxel `r` lands at `M @ after @ deskew @ camera @ pre @ r` (the mirror of views 2 and 3,
post-deskew flips, the loaded deskew, the camera alignment and pre-deskew flips), and records that matrix with the
stored view. `pick_impp_views.py` reads each factor from impp and refuses to solve unless they reproduce the recorded
matrix. impp loads a file's matrix rescaled to the plate's frame step and with the top arm's y shift taken out. The
solve writes matrices in the file's own frame anyway, so they can be pasted or assembled as they are. `impp_frames.py`
documents the model, and its test recovers known matrices from synthetic picks.

- **View fits:** `A_new = inv(POST) @ W @ POST @ A_old`, with `W` the world-space fit of B's beads onto A's, and
  `POST = T(origin) @ M @ after @ Sh @ F`.
- **Camera fits:** each pick is mapped back to B's raw frame, and the (Y, X) affine is solved there:
  `camera_new @ pre @ source = camera @ pre @ target`. This is the matrix impp applies, so no inversion is needed.
  The picks' raw frame (z) mismatch is reported: two cameras expose each frame together, so a frame mismatch is pick
  error.
- **Camera keys:** the camera key written is the one under which the file holds the matrix impp applied. impp may key
  cameras by dataset order rather than by `camera_name_to_id`.
