#
# Step traversability, ported from traversability_estimation_filters/src/StepFilter.cpp:
# https://github.com/leggedrobotics/traversability_estimation
#
# Two-pass algorithm, reproduced exactly (not a generic gradient approximation):
#
#   Pass 1 (first_window_radius): for every cell with any valid neighbour in the circular
#     window, step_height = max(height) - min(height) over that window.
#
#   Pass 2 (second_window_radius): for every cell, sweep step_height over the (generally
#     larger) second window and compute:
#       stepMax = max(step_height) found in the window
#       nCells  = count of neighbours whose step_height > critical_value
#       step    = min(stepMax, stepMax * nCells / critical_cell_number)
#     -- i.e. step is damped toward zero unless at least `critical_cell_number` cells in the
#     neighbourhood independently show an excessive height range, so a single noisy cell can't
#     trigger non-traversability on its own; once nCells >= critical_cell_number the damping
#     factor saturates at 1 and the full stepMax applies.
#
#   T_step = 1 - step/critical_value   if step < critical_value
#          = 0                          otherwise
#
# Deviation from upstream: upstream erases the intermediate "step_height" layer at the end of
# the filter. This pipeline publishes it (per the task spec's explicit visualization/debugging
# requirement), and also does not gate pass 1 on the center cell's own elevation validity the
# way upstream's outer loop nominally does -- upstream's own inner loop already determines
# validity independently of that outer check (see StepFilter.cpp:112-116, where `height` is
# immediately overwritten and unused), so this port gates purely on "does the window contain
# at least one valid sample", matching upstream's actual (not merely nominal) behaviour.
import cupy as cp
import numpy as np
from typing import List, Optional

from .plugin_manager import PluginBase


class StepTraversability(PluginBase):
    def __init__(
        self,
        cell_n: int = 200,
        resolution: float = 0.1,
        input_layer_name: str = "inpaint",
        critical_value: float = 0.06,
        first_window_radius: float = 0.15,
        second_window_radius: float = 0.25,
        critical_cell_number: int = 4,
        output: str = "traversability",  # "traversability" or "step_height"
        **kwargs,
    ):
        super().__init__()
        self.cell_n = int(cell_n)
        self.resolution = float(resolution)
        self.input_layer_name = input_layer_name
        self.critical_value = float(critical_value)
        self.critical_cell_number = max(1, int(critical_cell_number))
        if output not in ("traversability", "step_height"):
            raise ValueError(f"step_traversability: output must be 'traversability' or 'step_height', got {output!r}")
        self.output = output

        min_allowed_radius = 0.5 * self.resolution
        if first_window_radius < min_allowed_radius or second_window_radius < min_allowed_radius:
            print(
                f"[step_traversability] a configured window radius is smaller than half a "
                f"grid cell ({min_allowed_radius:.3f}m at resolution={self.resolution:.3f}m) "
                f"-- it would only ever see the center cell itself. Clamping up."
            )
            first_window_radius = max(first_window_radius, min_allowed_radius)
            second_window_radius = max(second_window_radius, min_allowed_radius)
        if second_window_radius < first_window_radius:
            print(
                f"[step_traversability] second_window_radius ({second_window_radius:.3f}m) is "
                f"smaller than first_window_radius ({first_window_radius:.3f}m) -- upstream's "
                f"design assumes the second pass aggregates over a broader area than the "
                f"first. Proceeding as configured, but this is likely a misconfiguration."
            )
        self.first_window_radius = float(first_window_radius)
        self.second_window_radius = float(second_window_radius)

        self._first_offsets = self._make_disk(self.first_window_radius)
        self._second_offsets = self._make_disk(self.second_window_radius)
        if len(self._second_offsets) < self.critical_cell_number:
            print(
                f"[step_traversability] WARNING: second_window_radius="
                f"{self.second_window_radius:.3f}m disk contains only "
                f"{len(self._second_offsets)} cells, fewer than critical_cell_number="
                f"{self.critical_cell_number}. nCells can never reach the critical count, so "
                f"the damping factor will always be < 1 -- traversability_step will never see "
                f"the full, undamped step value. Lower critical_cell_number or increase "
                f"second_window_radius."
            )

    def _make_disk(self, radius_m: float):
        max_cell_radius = int(np.ceil(radius_m / self.resolution))
        offsets = []
        for dr in range(-max_cell_radius, max_cell_radius + 1):
            for dc in range(-max_cell_radius, max_cell_radius + 1):
                y_off = dr * self.resolution
                x_off = dc * self.resolution
                if (x_off * x_off + y_off * y_off) <= (radius_m ** 2 + 1e-9):
                    offsets.append((dr, dc))
        return offsets

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

    def _lookup(self, elevation_map, layer_names, plugin_layers, plugin_layer_names) -> Optional[cp.ndarray]:
        name = self.input_layer_name
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
        height = self._lookup(elevation_map, layer_names, plugin_layers, plugin_layer_names)
        if height is None:
            return cp.full((self.cell_n, self.cell_n), cp.nan, dtype=cp.float32)
        height = height.astype(cp.float32)

        # Pass 1: local height range (max - min) within first_window_radius.
        height_max = cp.full_like(height, -cp.inf)
        height_min = cp.full_like(height, cp.inf)
        any_valid_1 = cp.zeros_like(height, dtype=bool)
        for dr, dc in self._first_offsets:
            z = self._shifted(height, dr, dc)
            valid = cp.isfinite(z)
            height_max = cp.where(valid, cp.maximum(height_max, z), height_max)
            height_min = cp.where(valid, cp.minimum(height_min, z), height_min)
            any_valid_1 |= valid
        step_height = cp.where(any_valid_1, height_max - height_min, cp.nan)

        if self.output == "step_height":
            return step_height.astype(cp.float32)

        # Pass 2: aggregate step_height over second_window_radius.
        step_max = cp.zeros_like(height)
        n_exceeding = cp.zeros_like(height)
        any_valid_2 = cp.zeros_like(height, dtype=bool)
        for dr, dc in self._second_offsets:
            sh = self._shifted(step_height, dr, dc)
            valid = cp.isfinite(sh)
            step_max = cp.where(valid, cp.maximum(step_max, cp.where(valid, sh, 0.0)), step_max)
            n_exceeding += cp.where(valid & (sh > self.critical_value), 1.0, 0.0)
            any_valid_2 |= valid

        damping = cp.clip(n_exceeding / float(self.critical_cell_number), 0.0, 1.0)
        step = step_max * damping  # equals min(stepMax, stepMax * nCells/nCellCritical)

        traversability = cp.clip(1.0 - step / self.critical_value, 0.0, 1.0)
        return cp.where(any_valid_2, traversability, cp.nan).astype(cp.float32)
