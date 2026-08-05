#
# Combined traversability, reproducing the upstream default fusion (robot_filter_parameter.yaml
# weightedSumFilter: MathExpressionFilter, "(1/3) * (slope + step + roughness)"):
#
#   T = (w_slope*T_slope + w_step*T_step + w_roughness*T_roughness) / (w_slope + w_step + w_roughness)
#
# which reduces to the plain equal-weight mean when all weights are 1.0 (the default, and the
# only combination upstream's own config actually uses).
#
#   terrain_cost = 1 - T
#
# Deviation from upstream: upstream's MathExpressionFilter just evaluates the arithmetic
# expression elementwise, so a single NaN input silently produces a NaN *result* for that
# cell -- which happens to already match the conservative behaviour this task requires (a
# missing component keeps the fused score unknown, not partially-averaged). This
# implementation makes that behaviour explicit and configurable via `allow_partial_average`
# (default False, matching upstream's actual per-cell result even though upstream itself never
# exposed the choice) rather than leaving it as an unstated side effect of NaN arithmetic.
import cupy as cp
from typing import List, Optional

from .plugin_manager import PluginBase


class TraversabilityFusion(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        slope_layer_name: str = "traversability_slope",
        step_layer_name: str = "traversability_step",
        roughness_layer_name: str = "traversability_roughness",
        slope_weight: float = 1.0,
        step_weight: float = 1.0,
        roughness_weight: float = 1.0,
        method: str = "mean",
        allow_partial_average: bool = False,
        output: str = "traversability",  # "traversability" or "terrain_cost"
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.slope_layer_name = slope_layer_name
        self.step_layer_name = step_layer_name
        self.roughness_layer_name = roughness_layer_name
        self.input_layer_names = [slope_layer_name, step_layer_name, roughness_layer_name]

        if method != "mean":
            raise ValueError(
                f"traversability_fusion: only fusion method 'mean' (weighted mean; equal "
                f"weights reproduce the upstream unweighted mean) is currently implemented, "
                f"got {method!r}."
            )
        self.method = method
        self.weights = cp.asarray(
            [float(slope_weight), float(step_weight), float(roughness_weight)], dtype=cp.float32
        )
        if float(self.weights.sum()) <= 0.0:
            raise ValueError("traversability_fusion: sum of slope/step/roughness weights must be > 0.")
        self.allow_partial_average = bool(allow_partial_average)
        if output not in ("traversability", "terrain_cost"):
            raise ValueError(f"traversability_fusion: output must be 'traversability' or 'terrain_cost', got {output!r}")
        self.output = output

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
        slope = self._lookup(self.slope_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        step = self._lookup(self.step_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        roughness = self._lookup(self.roughness_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        if slope is None or step is None or roughness is None:
            return cp.full((self.cell_n, self.cell_n), cp.nan, dtype=cp.float32)

        components = cp.stack([slope, step, roughness], axis=0)  # (3, cell_n, cell_n)
        valid = cp.isfinite(components)
        w = self.weights[:, None, None]

        if self.allow_partial_average:
            weighted_sum = cp.sum(cp.where(valid, components * w, 0.0), axis=0)
            weight_sum = cp.sum(cp.where(valid, w, 0.0), axis=0)
            have_any = weight_sum > 0.0
            traversability = cp.where(have_any, weighted_sum / cp.where(have_any, weight_sum, 1.0), cp.nan)
        else:
            all_valid = cp.all(valid, axis=0)
            filled = cp.where(valid, components, 0.0)
            weighted_sum = cp.sum(filled * w, axis=0)
            traversability = cp.where(all_valid, weighted_sum / float(self.weights.sum()), cp.nan)

        traversability = cp.clip(traversability, 0.0, 1.0)

        if self.output == "traversability":
            return traversability.astype(cp.float32)

        terrain_cost = cp.clip(1.0 - traversability, 0.0, 1.0)
        return cp.where(cp.isfinite(traversability), terrain_cost, cp.nan).astype(cp.float32)
