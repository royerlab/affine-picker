try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .ortho_view_manager import show_point_picker
from .ortho_view_widget import OrthoViewWidget
from .point_picker_widget import (
    FULL_RES_ORIGIN_KEY,
    PointPickerWidget,
    downsampled_layer_kwargs,
    estimate_affine_from_points,
    estimate_affine_from_points_components,
    estimate_affine_from_points_masked,
    estimate_affine_from_points_no_scale,
)

__all__ = (
    "FULL_RES_ORIGIN_KEY",
    "OrthoViewWidget",
    "PointPickerWidget",
    "downsampled_layer_kwargs",
    "estimate_affine_from_points",
    "estimate_affine_from_points_components",
    "estimate_affine_from_points_masked",
    "estimate_affine_from_points_no_scale",
    "show_point_picker",
)
