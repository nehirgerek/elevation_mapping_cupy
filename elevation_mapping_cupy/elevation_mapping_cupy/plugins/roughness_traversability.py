#
# Roughness traversability, ported from
# traversability_estimation_filters/src/RoughnessFilter.cpp:
# https://github.com/leggedrobotics/traversability_estimation
#
# For each cell with a valid surface normal, gather all valid height samples within
# `estimation_radius`, form the plane through their local mean point using the CELL'S OWN
# normal (not a plane refit to the neighbours), and compute the perpendicular distance of
# each sample from that plane:
#
#   plane_offset = normal . mean_point
#   dist_k       = normal . point_k - plane_offset
#   roughness    = sqrt( sum(dist_k^2) / (n - 1) )     <- sample stdev, Bessel-corrected (n-1)
#
# This is upstream's exact normalization (n-1, not n) -- documented here because it is easy to
# silently "fix" to a population stdev (n) when reimplementing; that would not match upstream.
#
#   T_roughness = 1 - roughness/critical_value   if roughness < critical_value
#               = 0                              otherwise
#
# Deviation from upstream: upstream has NO minimum-sample guard at all -- with n=1 valid
# neighbour it divides by (n-1)=0. This port adds the `minimum_valid_cells` requirement the
# task spec calls for; below that count both `roughness` and `traversability_roughness` are
# NaN rather than a spurious value.
import cupy as cp
import numpy as np
from typing import List, Optional

from .plugin_manager import PluginBase


class RoughnessTraversability(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        resolution: float = 0.1,
        input_layer_name: str = "inpaint",
        normal_x_layer_name: str = "surface_normal_x",
        normal_y_layer_name: str = "surface_normal_y",
        normal_z_layer_name: str = "surface_normal_z",
        estimation_radius: float = 0.15,
        critical_value: float = 0.04,
        minimum_valid_cells: int = 5,
        output: str = "traversability",  # "traversability" or "roughness" -- two YAML entries, one class
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.resolution = float(resolution)
        self.input_layer_name = input_layer_name
        self.normal_x_layer_name = normal_x_layer_name
        self.normal_y_layer_name = normal_y_layer_name
        self.normal_z_layer_name = normal_z_layer_name
        # Multi-dependency auto-chase (plugin_manager.py extension).
        self.input_layer_names = [input_layer_name, normal_x_layer_name, normal_y_layer_name, normal_z_layer_name]

        min_allowed_radius = 0.5 * self.resolution
        if estimation_radius < min_allowed_radius:
            print(
                f"[roughness_traversability] estimation_radius={estimation_radius:.3f}m is "
                f"smaller than half a grid cell ({min_allowed_radius:.3f}m). Clamping up."
            )
            estimation_radius = min_allowed_radius
        self.estimation_radius = float(estimation_radius)
        self.critical_value = float(critical_value)
        self.minimum_valid_cells = max(2, int(minimum_valid_cells))
        if output not in ("traversability", "roughness"):
            raise ValueError(f"roughness_traversability: output must be 'traversability' or 'roughness', got {output!r}")
        self.output = output

        max_cell_radius = int(np.ceil(self.estimation_radius / self.resolution))
        offsets = []
        for dr in range(-max_cell_radius, max_cell_radius + 1):
            for dc in range(-max_cell_radius, max_cell_radius + 1):
                y_off = dr * self.resolution
                x_off = dc * self.resolution
                if (x_off * x_off + y_off * y_off) <= (self.estimation_radius ** 2 + 1e-9):
                    offsets.append((dr, dc, x_off, y_off))
        if len(offsets) < self.minimum_valid_cells:
            print(
                f"[roughness_traversability] WARNING: disk at estimation_radius="
                f"{self.estimation_radius:.3f}m contains only {len(offsets)} cells, fewer than "
                f"minimum_valid_cells={self.minimum_valid_cells}. Every cell would be NaN."
            )
        self._offsets = offsets

    @staticmethod
    def _shifted(arr: cp.ndarray, dr: int, dc: int) -> cp.ndarray:
        out = cp.full_like(arr, cp.nan)
        h, w = arr.shape
        r_src_lo, r_src_hi = max(0, dr), min(h, h + dr)
        c_src_lo, c_src_hi = max(0, dc), min(w, w + dc)
        r_dst_lo, r_dst_hi = max(0, -dr), min(h, h - dr)
        c_dst_lo, c_dst_hi = max(0, -dc), min(w, w - dc)
        if r_src_hi > r_src_lo and c_src_hi > c_src_lo:
            out[r_dst_lo:r_dst_hi, c_dst_lo:c_dst_hi] = arr[r_src_lo:r_src_hi, c_src_lo:c_src_hi]
        return out

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
        height = self._lookup(self.input_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        nx = self._lookup(self.normal_x_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        ny = self._lookup(self.normal_y_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        nz = self._lookup(self.normal_z_layer_name, elevation_map, layer_names, plugin_layers, plugin_layer_names)
        if height is None or nx is None or ny is None or nz is None:
            return cp.full((self.cell_n, self.cell_n), cp.nan, dtype=cp.float32)
        height = height.astype(cp.float32)

        has_normal = cp.isfinite(nx) & cp.isfinite(ny) & cp.isfinite(nz)

        n_pts = cp.zeros_like(height)
        sx = cp.zeros_like(height)
        sy = cp.zeros_like(height)
        sz = cp.zeros_like(height)
        for dr, dc, x_off, y_off in self._offsets:
            z = self._shifted(height, dr, dc)
            valid = cp.isfinite(z)
            zf = cp.where(valid, z, 0.0)
            w = valid.astype(cp.float32)
            n_pts += w
            sx += w * x_off
            sy += w * y_off
            sz += w * zf

        enough = has_normal & (n_pts >= self.minimum_valid_cells)
        n_safe = cp.where(enough, n_pts, 1.0)
        mean_x = sx / n_safe
        mean_y = sy / n_safe
        mean_z = sz / n_safe

        # Second pass: perpendicular distance of every valid neighbour to the plane through
        # (mean_x, mean_y, mean_z) with this cell's own normal.
        sum_sq = cp.zeros_like(height)
        for dr, dc, x_off, y_off in self._offsets:
            z = self._shifted(height, dr, dc)
            valid = cp.isfinite(z)
            zf = cp.where(valid, z, 0.0)
            dist = nx * (x_off - mean_x) + ny * (y_off - mean_y) + nz * (zf - mean_z)
            sum_sq += cp.where(valid, dist * dist, 0.0)

        denom = cp.where(enough, n_pts - 1.0, 1.0)
        roughness = cp.sqrt(sum_sq / denom)
        roughness = cp.where(enough, roughness, cp.nan)

        if self.output == "roughness":
            return roughness.astype(cp.float32)

        traversability = cp.clip(1.0 - roughness / self.critical_value, 0.0, 1.0)
        return cp.where(enough, traversability, cp.nan).astype(cp.float32)
