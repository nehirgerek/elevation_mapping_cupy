"""Tests for the stateful, world-aligned occupancy plugin (persistent_occupancy.py).

Unlike the other plugins in this pipeline, persistent_occupancy.py is NOT a pure function of
its inputs -- it holds its own confidence/state arrays across calls. These tests exercise it
directly (construct one instance, call it repeatedly with varying synthetic inputs) rather
than through a fresh PluginManager per test, since the whole point is to verify behavior
ACROSS cycles.
"""
import cupy as cp
import numpy as np
import pytest

from elevation_mapping_cupy.plugins.persistent_occupancy import PersistentOccupancy, OCCUPIED, FREE, UNKNOWN

CORE_LAYER_NAMES = ["elevation", "variance", "is_valid", "traversability", "time", "upper_bound", "is_upper_bound"]


def _make_plugin(cell_n=20, **overrides):
    params = dict(
        cell_n=cell_n,
        resolution=0.1,
        traversability_layer_name="geom_traversability",
        measured_validity_layer="elevation",
        variance_layer="variance",
        lethal_traversability_threshold=0.30,
        maximum_variance=20.0,
        occupied_increment=2.0,
        free_decrement=1.0,
        occupied_threshold=2.0,
        free_threshold=-2.0,
        minimum_confidence=-6.0,
        maximum_confidence=6.0,
    )
    params.update(overrides)
    params.pop("resolution", None)  # PersistentOccupancy doesn't take resolution
    return PersistentOccupancy(**params)


def _frame(cell_n, elevation=0.0, variance=0.05, is_valid=1.0, traversability=1.0):
    """Build one cycle's (core elevation_map, plugin_layers) pair, uniform-valued for
    simplicity -- individual tests overwrite specific cells where they need variation."""
    m = cp.zeros((7, cell_n, cell_n), dtype=cp.float32)
    m[0] = elevation
    m[1] = variance
    m[2] = is_valid
    plugin_layers = cp.full((1, cell_n, cell_n), traversability, dtype=cp.float32)
    plugin_layer_names = ["geom_traversability"]
    return m, plugin_layers, plugin_layer_names


def _call(plugin, cell_n, **frame_kwargs):
    m, plugin_layers, plugin_layer_names = _frame(cell_n, **frame_kwargs)
    return plugin(m, CORE_LAYER_NAMES, plugin_layers, plugin_layer_names)


# ---------------------------------------------------------------------------
# 1. A brand new cell starts unknown with neutral confidence.
# ---------------------------------------------------------------------------
def test_initial_state_is_unknown_and_neutral():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    assert bool(cp.all(plugin._state == UNKNOWN))
    assert bool(cp.all(plugin._confidence == 0.0))


# ---------------------------------------------------------------------------
# 2. A single confidently-hazardous cycle is enough to mark a cell occupied at the default
# params (occupied_increment == occupied_threshold == 2.0).
# ---------------------------------------------------------------------------
def test_single_hazard_reading_marks_occupied():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    out = _call(plugin, cell_n, traversability=0.10, variance=0.05, is_valid=1.0)  # confident hazard
    assert bool(cp.all(out == OCCUPIED))
    assert bool(cp.all(plugin._confidence == 2.0))


# ---------------------------------------------------------------------------
# 3. Occupied persists through a cycle with no data at all (not observed this cycle).
# ---------------------------------------------------------------------------
def test_occupied_persists_through_missing_observation():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    _call(plugin, cell_n, traversability=0.10, variance=0.05, is_valid=1.0)
    assert bool(cp.all(plugin._state == OCCUPIED))

    out = _call(plugin, cell_n, traversability=cp.nan, variance=1000.0, is_valid=0.0)
    assert bool(cp.all(out == OCCUPIED)), "a cell must not lose its occupied state just because it wasn't observed this cycle"
    assert bool(cp.all(plugin._confidence == 2.0)), "confidence must not decay from lack of observation"


# ---------------------------------------------------------------------------
# 4. Occupied persists through a cycle that's merely uncertain (high variance, not a
# confirmed hazard) -- high variance alone must never count as free evidence.
# ---------------------------------------------------------------------------
def test_occupied_persists_through_high_variance_non_hazard_reading():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    _call(plugin, cell_n, traversability=0.10, variance=0.05, is_valid=1.0)
    assert bool(cp.all(plugin._state == OCCUPIED))

    # Observed, not a hazard by score, but variance far exceeds maximum_variance -- not
    # "reliably" free, so this must contribute neither occupied nor free evidence.
    out = _call(plugin, cell_n, traversability=0.90, variance=100.0, is_valid=1.0)
    assert bool(cp.all(out == OCCUPIED))
    assert bool(cp.all(plugin._confidence == 2.0))


# ---------------------------------------------------------------------------
# 5. Clearing an occupied cell requires multiple RELIABLE free reobservations, not one.
# ---------------------------------------------------------------------------
def test_clearing_occupied_cell_requires_multiple_reliable_free_cycles():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    # Drive confidence to the maximum via repeated hazard readings.
    for _ in range(5):
        _call(plugin, cell_n, traversability=0.10, variance=0.05, is_valid=1.0)
    assert float(plugin._confidence[0, 0]) == pytest.approx(6.0)  # clamped at maximum_confidence
    assert bool(cp.all(plugin._state == OCCUPIED))

    # From confidence=6.0, free_decrement=1.0, free_threshold=-2.0: need (6 - (-2)) = 8 reliable
    # free cycles before it flips. Check it stays OCCUPIED for all but the last of those.
    for i in range(7):
        out = _call(plugin, cell_n, traversability=0.90, variance=0.05, is_valid=1.0)
        assert bool(cp.all(out == OCCUPIED)), f"flipped to free too early, at reliable-free cycle {i + 1}/8"

    out = _call(plugin, cell_n, traversability=0.90, variance=0.05, is_valid=1.0)
    assert bool(cp.all(out == FREE)), "should have flipped to free on the 8th reliable-free cycle"
    assert float(plugin._confidence[0, 0]) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 6. A cell with no prior history needs 2 reliable-free cycles to actually confirm FREE
# (confidence must cross free_threshold=-2.0 from neutral 0.0), staying UNKNOWN in between --
# it must never jump straight from UNKNOWN to FREE on a single reading.
# ---------------------------------------------------------------------------
def test_new_cell_needs_two_free_cycles_to_confirm_free_not_one():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    out = _call(plugin, cell_n, traversability=0.90, variance=0.05, is_valid=1.0)
    assert bool(cp.all(out == UNKNOWN)), "a single free reading must not immediately confirm free"
    out = _call(plugin, cell_n, traversability=0.90, variance=0.05, is_valid=1.0)
    assert bool(cp.all(out == FREE))


# ---------------------------------------------------------------------------
# 7. on_map_shift rolls both arrays and pads newly-exposed cells to neutral/unknown.
# ---------------------------------------------------------------------------
def test_on_map_shift_rolls_and_pads_correctly():
    cell_n = 6
    plugin = _make_plugin(cell_n)
    plugin._confidence[:] = cp.arange(cell_n * cell_n, dtype=cp.float32).reshape(cell_n, cell_n)
    plugin._state[:] = OCCUPIED

    shift_value = cp.array([2, -1], dtype=cp.int32)  # shift rows +2, cols -1
    expected_confidence = cp.roll(plugin._confidence.copy(), shift_value, axis=(0, 1))
    expected_confidence[:2, :] = 0.0  # rows exposed by a +2 row shift
    expected_confidence[:, -1:] = 0.0  # cols exposed by a -1 col shift

    plugin.on_map_shift(shift_value)

    assert bool(cp.all(plugin._confidence == expected_confidence))
    assert bool(cp.all(plugin._state[:2, :] == UNKNOWN))
    assert bool(cp.all(plugin._state[:, -1:] == UNKNOWN))
    assert bool(cp.all(plugin._state[2:, :-1] == OCCUPIED))  # untouched region kept its value


# ---------------------------------------------------------------------------
# 8. on_map_clear resets everything to the initial neutral/unknown defaults.
# ---------------------------------------------------------------------------
def test_on_map_clear_resets_state():
    cell_n = 10
    plugin = _make_plugin(cell_n)
    _call(plugin, cell_n, traversability=0.10, variance=0.05, is_valid=1.0)
    assert bool(cp.all(plugin._state == OCCUPIED))

    plugin.on_map_clear()
    assert bool(cp.all(plugin._state == UNKNOWN))
    assert bool(cp.all(plugin._confidence == 0.0))


# ---------------------------------------------------------------------------
# 9. Parameter validation.
# ---------------------------------------------------------------------------
def test_invalid_threshold_ordering_raises():
    with pytest.raises(ValueError):
        _make_plugin(cell_n=5, free_threshold=3.0, occupied_threshold=2.0)
    with pytest.raises(ValueError):
        _make_plugin(cell_n=5, minimum_confidence=1.0, maximum_confidence=-1.0)
