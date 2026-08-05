#
# Tri-state occupancy classification for Mighty, computed as a plugin layer holding the
# standard nav_msgs/OccupancyGrid values directly (-1 unknown, 0 free, 100 lethal) as float32;
# elevation_mapping_node.py casts this layer to int8 when building the actual OccupancyGrid
# message. This file does grid-cell classification only, not message construction.
#
#   if not originally_observed or not finite(traversability) or variance > maximum_variance:
#       occupancy = -1   (unknown)
#   elif traversability < lethal_traversability_threshold:
#       occupancy = 100  (hard non-traversable)
#   else:
#       occupancy = 0    (traversable)
#
# "originally_observed" is the measured-validity mask (core `measured_validity_layer` layer
# finite AND core is_valid > 0.5) when `inpaint_only_is_unknown` is True (the default) --
# cells that only exist because the inpainting plugin filled a hole stay unknown here even
# though they have a finite, geometrically-usable height for the upstream stages. When
# `inpaint_only_is_unknown` is False, any finite `input_elevation_layer` cell counts as
# observed (less conservative; not the default).
#
# No continuous cost is ever encoded in the 1-99 range -- only -1/0/100, per the task's
# explicit requirement that Mighty does not treat those as gradual terrain cost.
import cupy as cp
from typing import List, Optional

from .plugin_manager import PluginBase


class TraversabilityOccupancy(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        traversability_layer_name: str = "geom_traversability",
        input_elevation_layer: str = "inpaint",
        measured_validity_layer: str = "elevation",
        variance_layer: str = "variance",
        maximum_variance: float = 10.0,
        lethal_traversability_threshold: float = 0.30,
        inpaint_only_is_unknown: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.traversability_layer_name = traversability_layer_name
        self.input_elevation_layer = input_elevation_layer
        self.measured_validity_layer = measured_validity_layer
        self.variance_layer = variance_layer
        self.input_layer_names = [traversability_layer_name, input_elevation_layer, measured_validity_layer, variance_layer]

        self.maximum_variance = float(maximum_variance)
        self.lethal_traversability_threshold = float(lethal_traversability_threshold)
        self.inpaint_only_is_unknown = bool(inpaint_only_is_unknown)

    @staticmethod
    def _lookup(name, elevation_map, layer_names, plugin_layers, plugin_layer_names) -> Optional[cp.ndarray]:
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
        traversability = self._lookup(
            self.traversability_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names
        )
        variance = self._lookup(self.variance_layer, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        measured = self._lookup(
            self.measured_validity_layer, elevation_map, layer_names, plugin_layers, plugin_layer_names
        )
        is_valid = elevation_map[layer_names.index("is_valid")] if "is_valid" in layer_names else None

        if traversability is None or variance is None or measured is None or is_valid is None:
            return cp.full((self.cell_n, self.cell_n), -1.0, dtype=cp.float32)

        if self.inpaint_only_is_unknown:
            originally_observed = cp.isfinite(measured) & (is_valid > 0.5)
        else:
            inpainted = self._lookup(
                self.input_elevation_layer, elevation_map, layer_names, plugin_layers, plugin_layer_names
            )
            originally_observed = cp.isfinite(inpainted) if inpainted is not None else cp.isfinite(measured)

        unknown = (~originally_observed) | (~cp.isfinite(traversability)) | (variance > self.maximum_variance)
        lethal = (~unknown) & (traversability < self.lethal_traversability_threshold)

        occupancy = cp.where(unknown, -1.0, cp.where(lethal, 100.0, 0.0))
        return occupancy.astype(cp.float32)
