#
# Stateful, world-aligned occupancy: replaces traversability_occupancy.py's purely stateless
# per-cycle classification (which rebuilds -1/0/100 from scratch every time it's called, so a
# single missing/uncertain/high-variance observation instantly overwrites whatever was known
# before) with a persistent per-cell confidence score that only moves in response to DIRECT
# evidence, and a hysteresis-resolved stored state that survives everything else:
#
#   - temporarily missing data (cell not currently observed this cycle)
#   - NaN geom_traversability (insufficient geometric support that cycle)
#   - high variance that isn't backed by a confident hazard reading
#
# No time-based decay term exists anywhere in this file -- a cell's confidence is unchanged
# unless THIS cycle produced direct occupied or free evidence for it. Passive drift (e.g. the
# core ElevationMap's own time_variance growth) can still eventually push a cell's variance
# above maximum_variance, but that alone no longer erases anything here: it just means this
# cycle contributes neither occupied nor free evidence, so confidence holds exactly where it
# was.
#
# Evidence model (per cell, per cycle):
#   occupied_evidence = originally_observed AND confidently hazardous this cycle
#     (geom_traversability < lethal_traversability_threshold, OR a configured
#     neighbor_step_hazard confirms one -- deliberately NOT gated on variance, matching this
#     pipeline's existing principle that a confident hazard reading should not be suppressed
#     just because the same cell's variance is also elevated -- see traversability_occupancy.py)
#   free_evidence = originally_observed AND confidently non-hazardous AND variance <=
#     maximum_variance (this direction IS variance-gated: the task this file implements is
#     explicit that clearing a dynamic object's former position requires the ground being
#     "directly and reliably" reobserved as free, not just weakly/uncertainly so)
#   Anything else (not observed, NaN traversability, or uncertain-and-not-hazardous) is
#   neither -- confidence is left untouched that cycle.
#
#   confidence = clip(
#       confidence + occupied_increment * occupied_evidence - free_decrement * free_evidence,
#       minimum_confidence, maximum_confidence,
#   )
#
# Hysteresis resolves the stored state from confidence, so a single ambiguous cycle can never
# flip an already-decided cell:
#   stored_state = OCCUPIED  if confidence >= occupied_threshold
#                = FREE      if confidence <= free_threshold
#                = <unchanged, i.e. previous stored_state> otherwise
#
# A freshly-allocated cell (map-edge cell that just rolled into view, or the very first cycle)
# starts at confidence=0.0 (neutral -- strictly between free_threshold and occupied_threshold
# at the documented defaults) and stored_state=UNKNOWN, matching the required "new cells begin
# unknown with neutral confidence."
#
# Persistence and the rolling map: this plugin holds its OWN confidence/state arrays as
# instance attributes (NOT as a PluginManager-managed layer -- those get wiped every cycle by
# PluginManager.reset_layers(), which would defeat the entire point). To stay spatially aligned
# as the map rolls, it implements PluginBase.on_map_shift (called by
# ElevationMap.shift_map_xy via PluginManager.notify_shift with the exact same [row_shift,
# col_shift] used to cp.roll the core elevation_map) and PluginBase.on_map_clear (called by
# ElevationMap.clear() via PluginManager.notify_clear). Newly-exposed cells after a shift are
# padded to the same neutral/unknown defaults as a freshly-allocated cell.
import cupy as cp
from typing import List, Optional

from .plugin_manager import PluginBase

OCCUPIED = 100.0
FREE = 0.0
UNKNOWN = -1.0


class PersistentOccupancy(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        traversability_layer_name: str = "geom_traversability",
        measured_validity_layer: str = "elevation",
        variance_layer: str = "variance",
        neighbor_step_hazard_layer_name: str = "",
        lethal_traversability_threshold: float = 0.30,
        maximum_variance: float = 20.0,
        occupied_increment: float = 2.0,
        free_decrement: float = 1.0,
        occupied_threshold: float = 2.0,
        free_threshold: float = -2.0,
        minimum_confidence: float = -6.0,
        maximum_confidence: float = 6.0,
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.traversability_layer_name = traversability_layer_name
        self.measured_validity_layer = measured_validity_layer
        self.variance_layer = variance_layer
        self.neighbor_step_hazard_layer_name = neighbor_step_hazard_layer_name
        self.input_layer_names = [traversability_layer_name]
        if neighbor_step_hazard_layer_name:
            self.input_layer_names.append(neighbor_step_hazard_layer_name)

        self.lethal_traversability_threshold = float(lethal_traversability_threshold)
        self.maximum_variance = float(maximum_variance)
        self.occupied_increment = float(occupied_increment)
        self.free_decrement = float(free_decrement)
        self.occupied_threshold = float(occupied_threshold)
        self.free_threshold = float(free_threshold)
        self.minimum_confidence = float(minimum_confidence)
        self.maximum_confidence = float(maximum_confidence)
        if self.minimum_confidence >= self.maximum_confidence:
            raise ValueError(
                f"persistent_occupancy: minimum_confidence ({self.minimum_confidence}) must be "
                f"< maximum_confidence ({self.maximum_confidence})"
            )
        if not (self.minimum_confidence <= self.free_threshold < self.occupied_threshold <= self.maximum_confidence):
            raise ValueError(
                "persistent_occupancy: require minimum_confidence <= free_threshold < "
                f"occupied_threshold <= maximum_confidence, got minimum_confidence="
                f"{self.minimum_confidence}, free_threshold={self.free_threshold}, "
                f"occupied_threshold={self.occupied_threshold}, maximum_confidence="
                f"{self.maximum_confidence}"
            )

        # Persistent state -- deliberately plain instance attributes, not a PluginManager
        # layer, so PluginManager.reset_layers() (called every point-cloud cycle AND every map
        # shift) cannot wipe them. See on_map_shift/on_map_clear for how they stay correct
        # across the two events that DO need to touch them.
        self._confidence = cp.zeros((self.cell_n, self.cell_n), dtype=cp.float32)
        self._state = cp.full((self.cell_n, self.cell_n), UNKNOWN, dtype=cp.float32)

    @staticmethod
    def _lookup(name, elevation_map, layer_names, plugin_layers, plugin_layer_names) -> Optional[cp.ndarray]:
        if name in layer_names:
            return elevation_map[layer_names.index(name)]
        if name in plugin_layer_names:
            return plugin_layers[plugin_layer_names.index(name)]
        return None

    def on_map_shift(self, shift_value) -> None:
        self._confidence = cp.roll(self._confidence, shift_value, axis=(0, 1))
        self._state = cp.roll(self._state, shift_value, axis=(0, 1))
        self._pad(self._confidence, shift_value, 0.0)
        self._pad(self._state, shift_value, UNKNOWN)

    def on_map_clear(self) -> None:
        self._confidence.fill(0.0)
        self._state.fill(UNKNOWN)

    @staticmethod
    def _pad(x, shift_value, value) -> None:
        """Pad the region exposed by cp.roll(x, shift_value, axis=(0,1)) with `value`,
        mirroring ElevationMap.pad_value's logic for a plain 2D (no leading layer axis) array."""
        if shift_value[0] > 0:
            x[: shift_value[0], :] = value
        elif shift_value[0] < 0:
            x[shift_value[0] :, :] = value
        if shift_value[1] > 0:
            x[:, : shift_value[1]] = value
        elif shift_value[1] < 0:
            x[:, shift_value[1] :] = value

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        traversability = self._lookup(
            self.traversability_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names
        )
        variance = self._lookup(self.variance_layer, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        measured = self._lookup(
            self.measured_validity_layer, elevation_map, layer_names, plugin_layers, plugin_layer_names
        )
        is_valid = elevation_map[layer_names.index("is_valid")] if "is_valid" in layer_names else None

        if traversability is None or variance is None or measured is None or is_valid is None:
            # Can't evaluate any evidence this cycle -- leave persistent state untouched
            # (no time decay, and no reason to erase existing knowledge over a wiring gap).
            return self._state.astype(cp.float32)

        originally_observed = cp.isfinite(measured) & (is_valid > 0.5)
        traversability_finite = cp.isfinite(traversability)

        confirmed_hazard = originally_observed & traversability_finite & (
            traversability < self.lethal_traversability_threshold
        )
        if self.neighbor_step_hazard_layer_name:
            hazard = self._lookup(
                self.neighbor_step_hazard_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names
            )
            if hazard is not None:
                confirmed_hazard = confirmed_hazard | (originally_observed & cp.isfinite(hazard) & (hazard > 0.5))

        reliable_free = (
            originally_observed
            & traversability_finite
            & (traversability >= self.lethal_traversability_threshold)
            & (variance <= self.maximum_variance)
        )

        confidence = self._confidence + self.occupied_increment * confirmed_hazard.astype(cp.float32)
        confidence = confidence - self.free_decrement * reliable_free.astype(cp.float32)
        confidence = cp.clip(confidence, self.minimum_confidence, self.maximum_confidence)

        state = cp.where(
            confidence >= self.occupied_threshold,
            OCCUPIED,
            cp.where(confidence <= self.free_threshold, FREE, self._state),
        )

        self._confidence = confidence.astype(cp.float32)
        self._state = state.astype(cp.float32)
        return self._state
