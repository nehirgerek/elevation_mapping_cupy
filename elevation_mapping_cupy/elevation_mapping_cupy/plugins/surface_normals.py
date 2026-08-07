#
# Geometric surface-normal estimator.
#
# Ported from grid_map's NormalVectorsFilter "area" (PCA) method:
# https://github.com/ANYbotics/grid_map/blob/master/grid_map_filters/src/NormalVectorsFilter.cpp
#
# For each eligible cell, gathers all valid neighbours within `estimation_radius` (a metric
# radius converted to a fixed disk of grid-cell offsets using the map resolution), computes
# their 3x3 covariance matrix, and takes the eigenvector of the smallest eigenvalue as the
# local plane normal -- the least-variance direction of a locally planar point set. This is
# the exact upstream mathematics (mean, covariance = E[pp^T] - mean*mean^T, eigendecomposition,
# smallest eigenvalue's eigenvector, sign-flipped toward +Z). Batched across the whole grid at
# once via cp.linalg.eigh over a stacked (cell_n, cell_n, 3, 3) covariance tensor -- no per-cell
# Python loop; the only Python loop is over the small, fixed set of disk stencil offsets
# (independent of grid size).
#
# Deviation from upstream: upstream falls back to a vertical normal (0,0,1) when fewer than 3
# points are available or the covariance matrix is degenerate (collinear points -- second
# eigenvalue ~0). This port instead produces NaN in both cases, per this pipeline's explicit
# requirement that a normal must never be silently invented from insufficient geometry --
# downstream slope/roughness stages must see "unknown" here, not "flat ground".
#
# The plugin framework registers one plugin instance per output layer (one 2D array per YAML
# block), so this same class is configured three times (component: x/y/z, matching the
# `type:` override convention already used by positive_spike_filter_cleanup in
# plugin_config.yaml) rather than returning all three axes from one call.
#
# PERFORMANCE: measured live, cp.linalg.eigh over the batched (cell_n,cell_n,3,3) covariance
# tensor is the dominant cost of this plugin (~30ms of ~30ms total at cell_n=400) -- NOT
# negligible when redundantly repeated 3x (once per x/y/z instance) every cycle. The three
# instances share the exact same input/estimation_radius/minimum_valid_cells in this pipeline's
# config, so they'd compute an IDENTICAL covariance tensor and eigendecomposition. _shared_cache
# below (class-level, keyed on PluginManager's own generation counter via on_generation_bump)
# lets whichever of the three instances runs first in a cycle do the real work once; the other
# two just slice out their own component from the cached full result. Cache is cleared on
# __init__ (i.e. whenever a PluginManager is (re)constructed) so multiple independent managers
# in the same process (e.g. across unit tests) can never see each other's cached arrays.
import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
from typing import List, Optional

from .plugin_manager import PluginBase

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class SurfaceNormals(PluginBase):
    # Class-level (shared across all x/y/z instances in this process) cache -- see module
    # docstring "PERFORMANCE" note. Cleared on every __init__ to prevent cross-manager leakage.
    _shared_cache: dict = {}

    def __init__(
        self,
        cell_n: int = 200,
        resolution: float = 0.1,
        input_layer_name: str = "inpaint",
        estimation_radius: float = 0.15,
        minimum_valid_cells: int = 6,
        component: str = "z",
        **kwargs,
    ):
        super().__init__()
        SurfaceNormals._shared_cache.clear()
        self._current_generation = -1  # sentinel: "no on_generation_bump seen yet"
        self.input_layer_name = input_layer_name
        self.resolution = float(resolution)
        self.cell_n = int(cell_n)
        if component not in _AXIS_INDEX:
            raise ValueError(f"surface_normals: component must be one of x/y/z, got {component!r}")
        self.component = component

        min_allowed_radius = 0.5 * self.resolution
        if estimation_radius < min_allowed_radius:
            print(
                f"[surface_normals] estimation_radius={estimation_radius:.3f}m is smaller than "
                f"half a grid cell ({min_allowed_radius:.3f}m) at resolution={self.resolution:.3f}m -- "
                f"it would only ever see the center cell itself. Clamping up to {min_allowed_radius:.3f}m."
            )
            estimation_radius = min_allowed_radius
        self.estimation_radius = float(estimation_radius)
        self.minimum_valid_cells = max(3, int(minimum_valid_cells))

        # Fixed disk of (row_offset, col_offset, x_offset_m, y_offset_m) stencil points. Row=Y,
        # Col=X is this repo's convention (see kernels/custom_kernels.py normal_filter_kernel).
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
                f"[surface_normals] WARNING: disk at estimation_radius={self.estimation_radius:.3f}m "
                f"contains only {len(offsets)} cells on this {self.resolution:.3f}m-resolution map, "
                f"fewer than minimum_valid_cells={self.minimum_valid_cells}. Every cell would be NaN "
                f"-- increase estimation_radius or lower minimum_valid_cells."
            )
        self._offsets = offsets

        # Precomputed correlation kernels -- replace the old per-offset Python shift-loop
        # (9-25 sequential cp.full_like/slice/add ops per call) with a single cupyx.scipy.
        # ndimage.correlate call per accumulator. correlate(x, kernel, mode='constant',
        # cval=0.0) with kernel[center+dr, center+dc] = weight(dr,dc) is mathematically
        # IDENTICAL to summing weight(dr,dc)*shifted(x,dr,dc) over all disk offsets (verified
        # bit-exact, max abs diff 0.0, against the original shift-loop on both plain-count and
        # geometrically-weighted accumulators, including with NaN holes and boundary rows) --
        # this is a pure vectorization, not an approximation. Kernels for x_off/y_off-weighted
        # accumulators (sx, sy, sxx, syy, sxy) are purely geometric (depend only on disk shape/
        # resolution, never on cell content), so they're built once here, not per-call.
        max_cell_radius = int(np.ceil(self.estimation_radius / self.resolution))
        ksize = 2 * max_cell_radius + 1
        center = max_cell_radius
        k_ones = cp.zeros((ksize, ksize), dtype=cp.float32)
        k_x = cp.zeros((ksize, ksize), dtype=cp.float32)
        k_y = cp.zeros((ksize, ksize), dtype=cp.float32)
        k_xx = cp.zeros((ksize, ksize), dtype=cp.float32)
        k_yy = cp.zeros((ksize, ksize), dtype=cp.float32)
        k_xy = cp.zeros((ksize, ksize), dtype=cp.float32)
        for dr, dc, x_off, y_off in offsets:
            k_ones[center + dr, center + dc] = 1.0
            k_x[center + dr, center + dc] = x_off
            k_y[center + dr, center + dc] = y_off
            k_xx[center + dr, center + dc] = x_off * x_off
            k_yy[center + dr, center + dc] = y_off * y_off
            k_xy[center + dr, center + dc] = x_off * y_off
        self._k_ones = k_ones
        self._k_x = k_x
        self._k_y = k_y
        self._k_xx = k_xx
        self._k_yy = k_yy
        self._k_xy = k_xy

    def on_generation_bump(self, generation: int) -> None:
        self._current_generation = generation

    @staticmethod
    def _shifted(arr: cp.ndarray, dr: int, dc: int) -> cp.ndarray:
        """out[r, c] = arr[r + dr, c + dc]; out-of-bounds filled with NaN.

        No circular wrap across the map edge: by the time plugins run, the map's own
        cp.roll-based motion handling (elevation_mapping.py:shift_map_xy) has already made
        this a plain, non-wrapping grid for this cycle -- see inspection notes on circular
        buffer handling.
        """
        out = cp.full_like(arr, cp.nan)
        h, w = arr.shape
        r_src_lo, r_src_hi = max(0, dr), min(h, h + dr)
        c_src_lo, c_src_hi = max(0, dc), min(w, w + dc)
        r_dst_lo, r_dst_hi = max(0, -dr), min(h, h - dr)
        c_dst_lo, c_dst_hi = max(0, -dc), min(w, w - dc)
        if r_src_hi > r_src_lo and c_src_hi > c_src_lo:
            out[r_dst_lo:r_dst_hi, c_dst_lo:c_dst_hi] = arr[r_src_lo:r_src_hi, c_src_lo:c_src_hi]
        return out

    def _get_input(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> Optional[cp.ndarray]:
        if self.input_layer_name in layer_names:
            return elevation_map[layer_names.index(self.input_layer_name)]
        if self.input_layer_name in plugin_layer_names:
            return plugin_layers[plugin_layer_names.index(self.input_layer_name)]
        return None

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        cache_key = (self.input_layer_name, self.estimation_radius, self.minimum_valid_cells, self.cell_n)
        cached = SurfaceNormals._shared_cache.get(cache_key)
        if cached is not None and cached[0] == self._current_generation:
            result = cached[1]
            return result[..., _AXIS_INDEX[self.component]].astype(cp.float32)

        height = self._get_input(elevation_map, layer_names, plugin_layers, plugin_layer_names)
        if height is None:
            return cp.full((self.cell_n, self.cell_n), cp.nan, dtype=cp.float32)
        height = height.astype(cp.float32)

        valid = cp.isfinite(height)
        valid_f = valid.astype(cp.float32)
        masked_z = cp.where(valid, height, 0.0)
        masked_zz = cp.where(valid, height * height, 0.0)

        def _corr(x, k):
            return ndi.correlate(x, k, mode="constant", cval=0.0)

        n_pts = _corr(valid_f, self._k_ones)
        sx = _corr(valid_f, self._k_x)
        sy = _corr(valid_f, self._k_y)
        sxx = _corr(valid_f, self._k_xx)
        syy = _corr(valid_f, self._k_yy)
        sxy = _corr(valid_f, self._k_xy)
        sz = _corr(masked_z, self._k_ones)
        szz = _corr(masked_zz, self._k_ones)
        sxz = _corr(masked_z, self._k_x)
        syz = _corr(masked_z, self._k_y)

        enough = n_pts >= self.minimum_valid_cells
        n_safe = cp.where(enough, n_pts, 1.0)

        mean_x = sx / n_safe
        mean_y = sy / n_safe
        mean_z = sz / n_safe

        cxx = sxx / n_safe - mean_x * mean_x
        cyy = syy / n_safe - mean_y * mean_y
        czz = szz / n_safe - mean_z * mean_z
        cxy = sxy / n_safe - mean_x * mean_y
        cxz = sxz / n_safe - mean_x * mean_z
        cyz = syz / n_safe - mean_y * mean_z

        # (cell_n, cell_n, 3, 3) batched covariance tensor.
        cov = cp.stack(
            [
                cp.stack([cxx, cxy, cxz], axis=-1),
                cp.stack([cxy, cyy, cyz], axis=-1),
                cp.stack([cxz, cyz, czz], axis=-1),
            ],
            axis=-2,
        )

        # cp.linalg.eigh matches Eigen::SelfAdjointEigenSolver's convention: ascending
        # eigenvalues, eigenvectors as columns. Column 0 (smallest eigenvalue) is the normal.
        eigvals, eigvecs = cp.linalg.eigh(cov)
        normal = eigvecs[..., :, 0]

        norm_len = cp.linalg.norm(normal, axis=-1)
        norm_len = cp.where(norm_len > 1e-9, norm_len, 1.0)
        normal = normal / norm_len[..., None]

        # Degenerate check: upstream requires the second eigenvalue (ascending, index 1) to be
        # non-trivial; near-zero means the points are collinear (no well-defined plane).
        not_degenerate = eigvals[..., 1] > 1e-8
        valid_normal = enough & not_degenerate

        # Orient upward: flip sign so nz >= 0 (normal_vector_positive_axis = z, upstream default).
        flip = normal[..., 2] < 0.0
        normal = cp.where(flip[..., None], -normal, normal)

        result = cp.where(valid_normal[..., None], normal, cp.nan)
        SurfaceNormals._shared_cache[cache_key] = (self._current_generation, result)
        return result[..., _AXIS_INDEX[self.component]].astype(cp.float32)
