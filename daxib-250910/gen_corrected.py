import io, sys
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
sys.argv=["x"]
import load_zarr_views_for_registration as L
PROJ=Path("/hpc/projects/group.royer/imaging/dev/2025_09_12_daxi_Daxi_B")
SHAPE=(1337,2372,4501)
slots={v:L._infer_pipeline_slot(alignment_mode="view",project_folder=PROJ,transform_yaml=None,
        view_id=v,camera_id=3,view_shape_zyx=SHAPE,allow_identity_old_transform=False) for v in (0,1,2,3)}
Cpair={(0,1):np.array([[1.,0.0188275676,-0.0000210862,-14.8398731522],[0.,0.9996207615,0.001800653,3.7393959173],[0.,-0.0020639651,1.0003720992,-24.9097497283],[0,0,0,1]]),
       (2,3):np.array([[1.,0.0183396577,0.0005187435,-19.0134200254],[0.,1.0060364622,-0.0006136403,-0.0705797288],[0.,-0.0031690225,1.0004008039,34.306131757],[0,0,0,1]])}
class FM:
    def __init__(s,C):s.C=C
    def get_estimated_affine(s):return s.C
for (f,m) in [(0,1),(2,3)]:
    inf=L.TransformInference(fixed_camera_id=3,moving_camera_id=3,fixed_view_id=f,moving_view_id=m,
        fixed_slot=slots[f],moving_slot=slots[m],output_mode="shared-space",invert_estimated_affine=False,
        output_inverse=False,show_full_pipeline_matrices=False,shared_space_offset_mode="split-offset",
        fixed_display_affine=np.eye(4),moving_display_affine=np.eye(4),fixed_view_source="t",moving_view_source="t")
    buf=io.StringIO()
    with redirect_stdout(buf): L._print_yaml_ready_affine(FM(Cpair[(f,m)]),inf)
    t=buf.getvalue()
    print(t[t.index("view_transforms:"):t.index("[TransformYAML] Also")])
