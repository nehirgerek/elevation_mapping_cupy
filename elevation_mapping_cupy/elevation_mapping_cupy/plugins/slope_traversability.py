#
# Slope traversability, ported from traversability_estimation_filters/src/SlopeFilter.cpp:
# https://github.com/leggedrobotics/traversability_estimation
#
#   theta = acos(clamp(normal_z, -1, 1))
#   T = 1 - theta/critical_value   if theta < critical_value
#     = 0                          otherwise
#
# critical_value is in radians (upstream default 1.0 rad; this repo's config recommends a
# value tuned to this rover, see traversability_plugin_config.yaml).
#
# Deviation from upstream: cells with no valid normal (NaN normal_z) produce NaN here, not 0
# -- "no data" and "known to be untraversable" must stay distinguishable downstream (fusion,
# occupancy).
import cupy as cp
from typing import List, Optional

from .plugin_manager import PluginBase


class SlopeTraversability(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        normal_z_layer_name: str = "surface_normal_z",
        critical_value: float = 0.52,
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.normal_z_layer_name = normal_z_layer_name
        # Single-dependency auto-chase (existing plugin_manager mechanism).
        self.input_layer_name = normal_z_layer_name
        if not (0.0 < critical_value <= (cp.pi / 2)):
            print(
                f"[slope_traversability] critical_value={critical_value:.3f} rad is outside "
                f"(0, pi/2] -- upstream requires critical slope in this interval."
            )
        self.critical_value = float(critical_value)

    def _get_input(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> Optional[cp.ndarray]:
        if self.normal_z_layer_name in layer_names:
            return elevation_map[layer_names.index(self.normal_z_layer_name)]
        if self.normal_z_layer_name in plugin_layer_names:
            return plugin_layers[plugin_layer_names.index(self.normal_z_layer_name)]
        return None

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        normal_z = self._get_input(elevation_map, layer_names, plugin_layers, plugin_layer_names)
        if normal_z is None:
            return cp.full((self.cell_n, self.cell_n), cp.nan, dtype=cp.float32)

        valid = cp.isfinite(normal_z)
        clamped_nz = cp.clip(cp.where(valid, normal_z, 0.0), -1.0, 1.0)
        theta = cp.arccos(clamped_nz)
        traversability = cp.clip(1.0 - theta / self.critical_value, 0.0, 1.0)
        return cp.where(valid, traversability, cp.nan).astype(cp.float32)
