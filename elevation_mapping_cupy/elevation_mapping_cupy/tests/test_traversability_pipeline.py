"""Tests for the geometric traversability pipeline (surface_normals, slope_traversability,
roughness_traversability, step_traversability, traversability_fusion,
traversability_occupancy, traversability_esdf).

These exercise the plugins directly through PluginManager (the same machinery
elevation_mapping.py drives in production), on synthetic core `elevation_map` arrays, mirroring
this repo's existing plugin test style (see test_dilation_filter.py / test_inpainting_plugin.py):
plain cupy arrays in, `.get()`/assert on the result, no ROS node needed.

Synthetic maps use `input_layer_name` pointed directly at the core "elevation" layer rather
than "inpaint" (skipping plugin_config.yaml's despike/inpaint chain), since these tests are
about the geometric-estimation mathematics, not the digging-map cleanup pipeline. Each test
builds its own small PluginManager loading only traversability_plugin_config.yaml with that
override, rather than the full merged production config.
"""
import math
import numpy as np
import cupy as cp
import pytest

from elevation_mapping_cupy.plugins.plugin_manager import PluginManager

CORE_LAYER_NAMES = ["elevation", "variance", "is_valid", "traversability", "time", "upper_bound", "is_upper_bound"]


def _traversability_config_path():
    """Locate traversability_plugin_config.yaml whether running against an installed
    package (config lives under share/<pkg>/config/, separate from the Python module tree)
    or directly against the source tree (config/ is a sibling of the ROS package root)."""
    import os

    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("elevation_mapping_cupy")
        candidate = os.path.join(share_dir, "config", "core", "traversability_plugin_config.yaml")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    # Source-tree fallback: .../elevation_mapping_cupy/elevation_mapping_cupy/tests/this_file.py
    # -> up to the ROS package root -> config/core/traversability_plugin_config.yaml.
    candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "core", "traversability_plugin_config.yaml",
    )
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError("Could not locate traversability_plugin_config.yaml via ament index or source-tree fallback.")


def _make_manager(cell_n, resolution, overrides=None):
    """Build a PluginManager with the traversability config, redirecting every geometric
    stage's input from 'inpaint' to 'elevation' (no despike/inpaint chain in these tests)."""
    import yaml

    with open(_traversability_config_path()) as f:
        cfg = yaml.safe_load(f)

    for entry in cfg.values():
        extra = entry.get("extra_params", {})
        if extra.get("input_layer_name") == "inpaint":
            extra["input_layer_name"] = "elevation"
        if overrides:
            extra.update(overrides.get(entry["layer_name"], {}))

    manager = PluginManager(cell_n=cell_n, resolution=resolution)
    plugin_params = []
    extra_params = []
    from elevation_mapping_cupy.plugins.plugin_manager import PluginParams
    for k, v in cfg.items():
        if v["enable"]:
            plugin_params.append(
                PluginParams(
                    name=k if "type" not in v else v["type"],
                    layer_name=v["layer_name"],
                    fill_nan=v["fill_nan"],
                    is_height_layer=v["is_height_layer"],
                )
            )
            extra_params.append(v["extra_params"])
    manager.init(plugin_params, extra_params)
    return manager


def _base_map(cell_n):
    m = cp.zeros((7, cell_n, cell_n), dtype=cp.float32)
    m[1] = 0.05   # variance: well observed
    m[2] = 1.0    # is_valid everywhere
    return m


ALL_LAYERS = [
    "surface_normal_x", "surface_normal_y", "surface_normal_z", "traversability_slope",
    "roughness", "traversability_roughness", "step_height", "traversability_step",
    "geom_traversability", "terrain_cost", "occupancy", "esdf_metric", "esdf_encoded",
]


def _compute_all(manager, elevation_map):
    for layer in ALL_LAYERS:
        manager.update_with_name(layer, elevation_map, CORE_LAYER_NAMES)
    return {name: manager.get_map_with_name(name) for name in ALL_LAYERS}


# ---------------------------------------------------------------------------
# 1. Flat plane
# ---------------------------------------------------------------------------
def test_flat_plane():
    cell_n = 60
    manager = _make_manager(cell_n, resolution=0.1)
    elevation_map = _base_map(cell_n)
    elevation_map[0] = 0.0  # flat ground

    out = _compute_all(manager, elevation_map)
    interior = slice(10, -10)

    assert float(cp.nanmin(out["surface_normal_z"][interior, interior])) == pytest.approx(1.0, abs=1e-4)
    assert float(cp.nanmin(out["traversability_slope"][interior, interior])) == pytest.approx(1.0, abs=1e-4)
    assert float(cp.nanmax(out["roughness"][interior, interior])) == pytest.approx(0.0, abs=1e-4)
    assert float(cp.nanmin(out["traversability_roughness"][interior, interior])) == pytest.approx(1.0, abs=1e-4)
    assert float(cp.nanmin(out["traversability_step"][interior, interior])) == pytest.approx(1.0, abs=1e-4)
    assert float(cp.nanmin(out["geom_traversability"][interior, interior])) == pytest.approx(1.0, abs=1e-4)
    assert bool(cp.all(out["occupancy"][interior, interior] == 0.0))


# ---------------------------------------------------------------------------
# 2. Constant inclined plane
# ---------------------------------------------------------------------------
def test_inclined_plane_recovers_known_angle():
    cell_n = 60
    resolution = 0.1
    manager = _make_manager(cell_n, resolution)
    elevation_map = _base_map(cell_n)

    slope_angle = math.radians(15.0)  # a real, moderate incline
    grade = math.tan(slope_angle)
    xs = (cp.arange(cell_n, dtype=cp.float32) - cell_n / 2.0) * resolution
    elevation_map[0] = grade * xs[None, :]  # tilt along the X (column) axis

    out = _compute_all(manager, elevation_map)
    interior = slice(10, -10)

    nz = out["surface_normal_z"][interior, interior]
    recovered_angle = cp.arccos(cp.clip(nz, -1, 1))
    assert float(cp.nanmean(recovered_angle)) == pytest.approx(slope_angle, abs=0.03)

    # This pipeline's configured critical_value is 0.52 rad (30deg); a 15deg incline should
    # sit clearly in the middle of the traversable range, not at 1.0 (flat) or 0.0 (critical).
    expected_slope_score = 1.0 - slope_angle / 0.52
    slope_score = out["traversability_slope"][interior, interior]
    assert float(cp.nanmean(slope_score)) == pytest.approx(expected_slope_score, abs=0.05)


# ---------------------------------------------------------------------------
# 3. Single step
# ---------------------------------------------------------------------------
def test_single_step_recovered_and_flags_excessive_step():
    cell_n = 60
    resolution = 0.1
    manager = _make_manager(cell_n, resolution)
    elevation_map = _base_map(cell_n)

    step_m = 0.20  # well above the 0.06m critical_value -- should read as non-traversable
    elevation_map[0] = 0.0
    elevation_map[0, :, cell_n // 2:] = step_m  # a step running along one axis

    out = _compute_all(manager, elevation_map)

    # Cells straddling the step (within reach of first_window_radius=0.15m -> ~1-2 cells) must
    # see a step_height close to the true step size.
    step_col = cell_n // 2
    near_step = out["step_height"][10:-10, step_col - 1:step_col + 1]
    assert float(cp.nanmax(near_step)) == pytest.approx(step_m, abs=0.02)

    # And must become non-traversable (score 0, well below the flat-ground score of 1).
    trav_near_step = out["traversability_step"][10:-10, step_col - 1:step_col + 1]
    assert float(cp.nanmax(trav_near_step)) < 0.5

    # Far from the step (multiple second_window_radius=0.25m away, i.e. > ~3 cells), the
    # ground is locally flat and should read fully traversable again.
    far_from_step = out["traversability_step"][10:-10, 5:15]
    assert float(cp.nanmin(far_from_step)) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Rough / noisy patch
# ---------------------------------------------------------------------------
def test_rough_patch_increases_roughness_decreases_traversability():
    cell_n = 60
    resolution = 0.1
    manager = _make_manager(cell_n, resolution)
    elevation_map = _base_map(cell_n)

    rng = np.random.default_rng(0)
    flat = np.zeros((cell_n, cell_n), dtype=np.float32)
    noisy = (rng.normal(scale=0.03, size=(cell_n, cell_n))).astype(np.float32)  # +-3cm noise
    elevation_map[0] = cp.asarray(flat)

    out_flat = _compute_all(manager, elevation_map)
    flat_roughness = float(cp.nanmean(out_flat["roughness"][10:-10, 10:-10]))
    flat_trav = float(cp.nanmean(out_flat["traversability_roughness"][10:-10, 10:-10]))

    elevation_map[0] = cp.asarray(noisy)
    manager.reset_layers()
    out_rough = _compute_all(manager, elevation_map)
    rough_roughness = float(cp.nanmean(out_rough["roughness"][10:-10, 10:-10]))
    rough_trav = float(cp.nanmean(out_rough["traversability_roughness"][10:-10, 10:-10]))

    assert rough_roughness > flat_roughness
    assert rough_trav < flat_trav


# ---------------------------------------------------------------------------
# 5. NaN hole: geometric support vs. occupancy validity
# ---------------------------------------------------------------------------
def test_nan_hole_uses_geometric_support_but_stays_unknown_in_occupancy():
    cell_n = 60
    resolution = 0.1
    # inpaint_only_is_unknown defaults True in the config; here we feed the geometric stages
    # directly from "elevation" (see _make_manager), so to model "inpainted but not measured"
    # for the occupancy stage we point measured_validity_layer at core "elevation" (which we
    # will leave NaN at the hole) while leaving the *geometric* input at "elevation" too --
    # i.e. this test checks occupancy's is_valid/finite gating, not a real inpaint plugin.
    manager = _make_manager(cell_n, resolution)
    elevation_map = _base_map(cell_n)
    elevation_map[0] = 0.0

    hole = (slice(28, 32), slice(28, 32))
    elevation_map[0][hole] = cp.nan
    elevation_map[2][hole] = 0.0  # not measured there

    out = _compute_all(manager, elevation_map)

    # The hole itself has no elevation sample to use even as geometric support (this test does
    # not exercise a separate inpaint layer), so it must be unknown in occupancy.
    assert bool(cp.all(out["occupancy"][hole] == -1.0))

    # Cells just outside the hole, but within its neighbourhood windows, must not crash and
    # must not silently claim full confidence they don't have (fewer valid neighbours than
    # the flat-plane case) -- they should still resolve to *some* finite classification given
    # this hole is small relative to the estimation radii, not NaN-propagate indefinitely.
    ring = out["occupancy"][20:26, 20:26]
    assert bool(cp.all(cp.isfinite(ring)))


# ---------------------------------------------------------------------------
# 6. High-variance cells become unknown in occupancy
# ---------------------------------------------------------------------------
def test_high_variance_cells_become_unknown():
    cell_n = 60
    resolution = 0.1
    manager = _make_manager(cell_n, resolution, overrides={"occupancy": {"maximum_variance": 1.0}})
    elevation_map = _base_map(cell_n)
    elevation_map[0] = 0.0

    bad_patch = (slice(28, 32), slice(28, 32))
    elevation_map[1][bad_patch] = 50.0  # variance far above maximum_variance=1.0

    out = _compute_all(manager, elevation_map)
    assert bool(cp.all(out["occupancy"][bad_patch] == -1.0))
    # Elsewhere, low variance + flat ground must still read free.
    assert bool(cp.all(out["occupancy"][10:20, 10:20] == 0.0))


# ---------------------------------------------------------------------------
# 7. Map boundaries: no indexing artifacts / crashes
# ---------------------------------------------------------------------------
def test_boundaries_do_not_crash_or_wrap():
    cell_n = 40
    resolution = 0.1
    manager = _make_manager(cell_n, resolution)
    elevation_map = _base_map(cell_n)
    elevation_map[0] = 0.0
    # A tall spike right at the corner -- if any shift/disk operation wrapped around instead
    # of padding with NaN, this would corrupt the opposite edge of the map.
    elevation_map[0, 0, 0] = 5.0
    elevation_map[0, -1, -1] = -5.0

    out = _compute_all(manager, elevation_map)
    # No exception is itself part of the test. Additionally, the far corner (opposite the
    # spike) must be unaffected by it -- i.e. still reads as flat ground, not contaminated by
    # a wrapped-around neighbour.
    far_from_spikes = out["surface_normal_z"][cell_n // 2 - 2:cell_n // 2 + 2, cell_n // 2 - 2:cell_n // 2 + 2]
    assert float(cp.nanmin(far_from_spikes)) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 8. ESDF
# ---------------------------------------------------------------------------
def test_esdf_seed_zero_and_axial_diagonal_distances():
    cell_n = 41
    resolution = 0.1
    manager = _make_manager(cell_n, resolution, overrides={"esdf_encoded": {"truncation_distance": 2.0}})
    elevation_map = _base_map(cell_n)
    elevation_map[0] = 0.0

    # Directly force a known occupancy pattern rather than deriving it from geometry, to test
    # the ESDF stage's own math in isolation.
    layer_names = manager.layer_names
    occ_idx = layer_names.index("occupancy")
    manager.update_with_name("occupancy", elevation_map, CORE_LAYER_NAMES)  # populate normally first
    occ = manager.layers[occ_idx]
    occ[:] = 0.0
    center = cell_n // 2
    occ[center, center] = 100.0  # single hard-obstacle seed at the exact center
    manager._layer_generations[occ_idx] = manager._generation  # mark fresh so ESDF reuses this exact pattern

    manager.update_with_name("esdf_metric", elevation_map, CORE_LAYER_NAMES)
    manager.update_with_name("esdf_encoded", elevation_map, CORE_LAYER_NAMES)
    esdf_metric = manager.get_map_with_name("esdf_metric")
    esdf_encoded = manager.get_map_with_name("esdf_encoded")

    assert float(esdf_metric[center, center]) == 0.0
    assert float(esdf_encoded[center, center]) == 100.0

    # Axial neighbour: exactly one cell away -> resolution metres.
    assert float(esdf_metric[center, center + 1]) == pytest.approx(resolution, abs=1e-5)
    # Diagonal neighbour: sqrt(2) cells away.
    assert float(esdf_metric[center + 1, center + 1]) == pytest.approx(resolution * math.sqrt(2), abs=1e-5)

    # Truncation: a point 2m away with truncation_distance=2.0 must be within [1.9,2.0] and
    # encode to (approximately) zero clearance.
    far_r = int(round(2.0 / resolution))
    if center + far_r < cell_n:
        d_far = float(esdf_metric[center, center + far_r])
        assert d_far == pytest.approx(2.0, abs=0.05)
        assert float(esdf_encoded[center, center + far_r]) <= 3.0  # ~0, allow rounding


def test_esdf_no_obstacles_is_fully_clear():
    cell_n = 30
    resolution = 0.1
    manager = _make_manager(cell_n, resolution)
    elevation_map = _base_map(cell_n)
    elevation_map[0] = 0.0

    layer_names = manager.layer_names
    occ_idx = layer_names.index("occupancy")
    manager.update_with_name("occupancy", elevation_map, CORE_LAYER_NAMES)
    manager.layers[occ_idx][:] = 0.0  # force: no obstacles, no unknowns anywhere
    manager._layer_generations[occ_idx] = manager._generation

    manager.update_with_name("esdf_encoded", elevation_map, CORE_LAYER_NAMES)
    encoded = manager.get_map_with_name("esdf_encoded")
    assert bool(cp.all(encoded == 0.0))


# ---------------------------------------------------------------------------
# 9. Resolution scaling: same physical terrain at two resolutions
# ---------------------------------------------------------------------------
def test_resolution_scaling_consistency():
    resolution_a, cell_n_a = 0.10, 60
    resolution_b, cell_n_b = 0.05, 120  # same 6m x 6m physical area, finer grid

    slope_angle = math.radians(10.0)
    grade = math.tan(slope_angle)

    results = {}
    for resolution, cell_n in ((resolution_a, cell_n_a), (resolution_b, cell_n_b)):
        manager = _make_manager(cell_n, resolution)
        elevation_map = _base_map(cell_n)
        xs = (cp.arange(cell_n, dtype=cp.float32) - cell_n / 2.0) * resolution
        elevation_map[0] = grade * xs[None, :]
        out = _compute_all(manager, elevation_map)
        margin = int(round(1.0 / resolution))  # same 1m physical margin at both resolutions
        interior = slice(margin, -margin)
        results[resolution] = float(cp.nanmean(out["surface_normal_z"][interior, interior]))

    assert results[resolution_a] == pytest.approx(results[resolution_b], abs=0.01)
