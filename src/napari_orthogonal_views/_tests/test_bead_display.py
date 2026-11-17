import numpy as np
import pytest

from napari_orthogonal_views.bead_display import enhance_beads

SHAPE = (40, 96, 128)
BEADS = [(10, 20, 30), (25, 70, 100), (30, 40, 60), (15, 80, 20)]
BRIGHTNESS = [1500, 60000, 4000, 800]


def _bead_volume(seed=0):
    """Background sloping along y (100 to 300 counts), noise, and beads."""
    rng = np.random.default_rng(seed)
    y = np.linspace(100, 300, SHAPE[1])[None, :, None]
    volume = np.broadcast_to(y, SHAPE) + rng.normal(0, 15, SHAPE)
    zz, yy, xx = np.indices(SHAPE)
    for (z, yc, x), peak in zip(BEADS, BRIGHTNESS, strict=True):
        r2 = (zz - z) ** 2 + (yy - yc) ** 2 + (xx - x) ** 2
        volume = volume + peak * np.exp(-r2 / (2 * 1.2**2))
    return np.clip(volume, 0, 65535).astype(np.uint16)


def test_background_and_gradient_become_zero():
    enhanced, _ = enhance_beads(_bead_volume())
    assert enhanced.dtype == np.uint16 and enhanced.shape == SHAPE
    # Far from any bead, at the dim and the bright end of the gradient:
    # about 84% of Gaussian noise is under a floor 1 sigma above the median.
    for region in (enhanced[:, :8, 64:], enhanced[:, -8:, 64:]):
        assert np.mean(region == 0) > 0.75
        assert np.percentile(region, 99) < 50  # noise is 15 counts


def test_bead_positions_are_unchanged():
    raw = _bead_volume()
    enhanced, (low, high) = enhance_beads(raw)
    for z, y, x in BEADS:
        box = (slice(z - 3, z + 4), slice(y - 3, y + 4), slice(x - 3, x + 4))
        peak = np.unravel_index(np.argmax(enhanced[box]), enhanced[box].shape)
        assert tuple(int(p) for p in peak) == (3, 3, 3)
        assert enhanced[z, y, x] > 0
    assert low == 0 and high > 0


def test_dimmest_bead_stays_visible_under_contrast_limits():
    enhanced, (_, high) = enhance_beads(_bead_volume())
    z, y, x = BEADS[BRIGHTNESS.index(min(BRIGHTNESS))]
    # Fixed 0-65535 limits put an 800-count bead at about 1% of the range.
    assert enhanced[z, y, x] / high > 0.05


@pytest.mark.parametrize("factor", [4, (1.0, 4.0, 4.0)])
def test_level_factor_scales_the_background_block(factor):
    enhanced, _ = enhance_beads(_bead_volume(), level_factor=factor)
    assert enhanced.shape == SHAPE
    assert np.mean(enhanced[:, :8, 64:] == 0) > 0.75


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match="3D"):
        enhance_beads(np.zeros((4, 4)))


def test_despeckle_removes_isolated_voxels_and_keeps_beads():
    raw = _bead_volume()
    speckled, _ = enhance_beads(raw, despeckle_neighbours=0)
    clean, _ = enhance_beads(raw)
    background = (slice(None), slice(None, 8), slice(64, None))
    assert np.mean(speckled[background] > 0) > 0.1
    assert np.mean(clean[background] > 0) < 0.01
    for z, y, x in BEADS:  # every bead, the dimmest included, keeps its apex
        assert clean[z, y, x] == speckled[z, y, x] > 0


def test_despeckle_does_not_depend_on_slab_size():
    raw = _bead_volume()
    reference, _ = enhance_beads(raw, slab=SHAPE[0])
    for slab in (1, 3, 7):
        np.testing.assert_array_equal(enhance_beads(raw, slab=slab)[0], reference)


def test_nan_padding_and_inf_overflow():
    """Stored float16 views have NaN padding and inf where bright beads overflow."""
    raw = _bead_volume().astype(np.float16)
    raw[:, :, :24] = np.nan  # a padded wedge, whole blocks wide
    z, y, x = BEADS[1]
    raw[z, y, x] = np.inf
    enhanced, (_, high) = enhance_beads(raw)
    assert np.all(enhanced[:, :, :24] == 0)
    assert enhanced[z, y, x] == 65535
    for bz, by, bx in BEADS:  # every bead outside the padding survives
        if bx >= 24:
            assert enhanced[bz, by, bx] > 0
    assert np.mean(enhanced[:, :8, 64:] == 0) > 0.75
    assert np.isfinite(high)
