#
# Two-dimensional Euclidean signed... (actually unsigned, obstacle-side-only) distance field
# from the tri-state occupancy classification.
#
# Seeds are cells with occupancy == 100 (hard obstacle). If `unknown_is_obstacle` is True
# (the conservative default), occupancy == -1 (unknown) cells are seeded too, so unexplored
# space reads as zero clearance rather than being silently treated as free. Free cells
# (occupancy == 0) get their true Euclidean distance (in metres) to the nearest seed via
# cupyx.scipy.ndimage.distance_transform_edt -- already used elsewhere in this codebase
# (kernels/custom_kernels.py) for nearest-valid-cell lookups, reused here for its intended
# purpose. This computes true Euclidean distance, not Manhattan/chebyshev.
#
# Two outputs (one class, two YAML entries, matching the rest of this pipeline):
#   "esdf_metric"  -- unclamped true distance in metres (kept for debugging/visualization;
#                      "internal" metric layer the task asks to preserve where possible).
#   "esdf_encoded" -- the Mighty-compatible encoding:
#                        v = round(100 * (1 - min(d, d_max) / d_max))
#                      so 100 = zero clearance (at/inside an obstacle or unknown-as-obstacle
#                      cell), 0 = at or beyond the truncation distance d_max. This is an
#                      *encoded distance field*, not an occupancy probability grid -- values
#                      1-99 here mean "this many percent of the way to full clearance", never
#                      "this cell is N% likely occupied".
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt
from typing import List, Optional

from .plugin_manager import PluginBase


class TraversabilityEsdf(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        resolution: float = 0.1,
        occupancy_layer_name: str = "occupancy",
        unknown_is_obstacle: bool = True,
        truncation_distance: float = 1.5,
        output: str = "esdf_encoded",  # "esdf_encoded" or "esdf_metric"
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.resolution = float(resolution)
        self.occupancy_layer_name = occupancy_layer_name
        self.input_layer_name = occupancy_layer_name
        self.unknown_is_obstacle = bool(unknown_is_obstacle)
        if truncation_distance <= 0.0:
            raise ValueError("traversability_esdf: truncation_distance must be > 0.")
        self.truncation_distance = float(truncation_distance)
        if output not in ("esdf_encoded", "esdf_metric"):
            raise ValueError(f"traversability_esdf: output must be 'esdf_encoded' or 'esdf_metric', got {output!r}")
        self.output = output

    def _lookup(self, elevation_map, layer_names, plugin_layers, plugin_layer_names) -> Optional[cp.ndarray]:
        name = self.occupancy_layer_name
        if name in layer_names:
            return elevation_map[layer_names.index(name)]
        if name in plugin_layer_names:
            return plugin_layers[plugin_layer_names.index(name)]
        return None

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        occupancy = self._lookup(elevation_map, layer_names, plugin_layers, plugin_layer_names)
        if occupancy is None:
            fallback = 0.0 if self.output == "esdf_metric" else 100.0
            return cp.full((self.cell_n, self.cell_n), fallback, dtype=cp.float32)

        seed = occupancy >= 99.5  # occupancy == 100 (hard obstacle)
        if self.unknown_is_obstacle:
            seed = seed | (occupancy <= -0.5)  # occupancy == -1 (unknown)

        if not bool(cp.any(seed)):
            # No obstacle/unknown seeds anywhere on the map: nothing to measure distance to.
            # Treat every cell as at-or-beyond the truncation distance (maximally clear),
            # rather than dividing by zero or leaving an undefined distance transform.
            if self.output == "esdf_metric":
                return cp.full((self.cell_n, self.cell_n), cp.inf, dtype=cp.float32)
            return cp.zeros((self.cell_n, self.cell_n), dtype=cp.float32)

        # distance_transform_edt(input) computes, for every True cell in `input`, the
        # (cell-unit) distance to the nearest False cell. We want, for every cell, the
        # distance to the nearest seed -- i.e. invert the seed mask so seeds are False
        # (distance 0 there) and everything else is True.
        distance_cells = distance_transform_edt(~seed)
        distance_m = distance_cells.astype(cp.float32) * self.resolution
        # Seeds themselves are exactly zero (edt gives 0 at False cells by definition, but be
        # explicit and robust to any implementation edge case).
        distance_m = cp.where(seed, 0.0, distance_m)

        if self.output == "esdf_metric":
            return distance_m

        clamped = cp.minimum(distance_m, self.truncation_distance)
        encoded = cp.round(100.0 * (1.0 - clamped / self.truncation_distance))
        return cp.clip(encoded, 0.0, 100.0).astype(cp.float32)
