# Pick projects

One folder per bead store, holding everything picked on it: the picks, how each session was run, and the calibration entries each fit produced. Text files only, no image data. `pick_impp_views.py` creates and fills these; `pick_projects.py` lists, accepts and assembles them.

```
<store>/store.yaml                              store path, pyramid shapes, channel labels
<store>/sessions/<time>_<mode>_v<A>-v<B>_c<A>-c<B>/
    session.yaml                                command, views, channels, flips, estimator, base calibration
    picks.json                                  the picks, autosaved after every edit
    result.yaml                                 latest applied fit and the calibration entries it changes
<store>/calibration/<base>_picked.yaml          base calibration with the accepted entries
<store>/calibration/assembly.yaml               which session set which entry, and what was removed
```

Folder names are the store's parent folder and name plus a short hash of its full path, so every pyramid level of a store shares one folder. Coordinates and residuals are in full-resolution voxels.
