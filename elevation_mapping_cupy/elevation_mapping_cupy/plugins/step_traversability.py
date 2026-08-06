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
import cupyx.scipy.ndimage as ndi
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

        # PERFORMANCE: replace the per-offset Python shift-loop (9 offsets for a typical
        # first_window_radius, up to ~50 for second_window_radius=0.4m) with cupyx.scipy.
        # ndimage max/min/correlate filters -- a single fused GPU call each instead of N
        # sequential shift+compare ops. Verified bit-exact (max abs diff 0.0) against the
        # original shift-loop, including NaN holes and boundary rows, via:
        #   max/min over a disk footprint <-> maximum_filter/minimum_filter(x_with_+-inf_for_
        #     invalid, footprint=disk, mode='constant', cval=-+inf)
        #   count/sum over a disk <-> correlate(mask_or_value, ones_kernel, mode='constant',
        #     cval=0.0)
        self._first_footprint, self._first_ones_kernel = self._make_footprint_and_kernel(self._first_offsets)
        self._second_footprint, self._second_ones_kernel = self._make_footprint_and_kernel(self._second_offsets)

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
    def _make_footprint_and_kernel(offsets):
        max_r = max((abs(dr) for dr, dc in offsets), default=0)
        max_c = max((abs(dc) for dr, dc in offsets), default=0)
        r_radius = max(max_r, max_c)  # square kernel large enough to hold every offset
        ksize = 2 * r_radius + 1
        center = r_radius
        footprint = cp.zeros((ksize, ksize), dtype=bool)
        kernel_ones = cp.zeros((ksize, ksize), dtype=cp.float32)
        for dr, dc in offsets:
            footprint[center + dr, center + dc] = True
            kernel_ones[center + dr, center + dc] = 1.0
        return footprint, kernel_ones

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
        height_for_max = cp.where(cp.isfinite(height), height, -cp.inf)
        height_for_min = cp.where(cp.isfinite(height), height, cp.inf)
        height_max = ndi.maximum_filter(height_for_max, footprint=self._first_footprint, mode="constant", cval=-cp.inf)
        height_min = ndi.minimum_filter(height_for_min, footprint=self._first_footprint, mode="constant", cval=cp.inf)
        any_valid_1 = ndi.correlate(
            cp.isfinite(height).astype(cp.float32), self._first_ones_kernel, mode="constant", cval=0.0
        ) > 0
        step_height = cp.where(any_valid_1, height_max - height_min, cp.nan)

        if self.output == "step_height":
            return step_height.astype(cp.float32)

        # Pass 2: aggregate step_height over second_window_radius.
        sh_for_max = cp.where(cp.isfinite(step_height), step_height, -cp.inf)
        windowed_max = ndi.maximum_filter(sh_for_max, footprint=self._second_footprint, mode="constant", cval=-cp.inf)
        step_max = cp.maximum(windowed_max, 0.0)  # step_max starts at 0.0 when no valid neighbour exists
        exceeding = cp.isfinite(step_height) & (step_height > self.critical_value)
        n_exceeding = ndi.correlate(exceeding.astype(cp.float32), self._second_ones_kernel, mode="constant", cval=0.0)
        any_valid_2 = ndi.correlate(
            cp.isfinite(step_height).astype(cp.float32), self._second_ones_kernel, mode="constant", cval=0.0
        ) > 0

        damping = cp.clip(n_exceeding / float(self.critical_cell_number), 0.0, 1.0)
        step = step_max * damping  # equals min(stepMax, stepMax * nCells/nCellCritical)

        traversability = cp.clip(1.0 - step / self.critical_value, 0.0, 1.0)
        return cp.where(any_valid_2, traversability, cp.nan).astype(cp.float32)
