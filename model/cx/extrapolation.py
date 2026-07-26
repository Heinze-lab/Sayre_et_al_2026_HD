import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix

DEFAULT_OUTPUT_KERNEL_SIZE = 3
DEFAULT_INPUT_KERNEL_SIZE = 9

class KernelParameters:
    def __init__(self, size, columnar, symmetric=True):
        assert size % 2 == 1, "kernel widths must be odd"
        self.indices = []
        self.size = size
        self.columnar = columnar
        self.symmetric = symmetric
    
    def allocate_indices(self, next_index):
        size = self.size if not self.columnar else self.size - 1
        if self.symmetric:
            size = size - size//2
        end = next_index + size
        self.indices = list(range(next_index, end))
        return end
    
    def extract(self, theta):
        kernel = theta[self.indices]
        if self.symmetric and self.columnar:
            return Kernel(np.hstack([[1], kernel]), True, self.size)
        elif not self.symmetric and self.columnar:
            mid = len(kernel)//2
            return Kernel(np.hstack([kernel[:mid], [1], kernel[mid:]]), False, self.size)
        else:
            return Kernel(kernel, self.symmetric, self.size)

    def extract_with_map(self, theta_tail: np.ndarray, index_base: int = 0) -> tuple["Kernel", np.ndarray]:
        """Build kernel and per-entry global flat-theta index (-1 if fixed / not in theta).

        theta_tail is theta[kernel_offset:] and index_base is that kernel_offset so maps index the full theta vector.
        """
        kernel = theta_tail[self.indices]
        gidx = index_base + np.asarray(self.indices, dtype=np.int64)
        if self.symmetric and self.columnar:
            vals = np.concatenate([[1.0], kernel])
            gmap = np.concatenate([[-1], gidx])
            return Kernel(vals, True, self.size, global_map=gmap), gmap
        if not self.symmetric and self.columnar:
            mid = len(kernel) // 2
            vals = np.concatenate([kernel[:mid], [1.0], kernel[mid:]])
            gmap = np.concatenate([gidx[:mid], [-1], gidx[mid:]])
            return Kernel(vals, False, self.size, global_map=gmap), gmap
        vals = np.asarray(kernel, dtype=float)
        return Kernel(vals, self.symmetric, self.size, global_map=gidx), gidx


class Kernel:
    def __init__(self, kernel, symmetric, size, global_map: np.ndarray | None = None):
        self.kernel = np.asarray(kernel, dtype=float)
        self.symmetric = symmetric
        self.size = size
        if global_map is None:
            self.global_map = np.full(len(self.kernel), -1, dtype=np.int64)
        else:
            self.global_map = np.asarray(global_map, dtype=np.int64)

    def _sample_index(self, delta: int, flip: bool) -> int:
        if flip:
            delta = -delta
        if self.symmetric:
            index = abs(delta)
        else:
            index = delta + len(self.kernel) // 2
        return index

    def sample(self, delta, flip=False):
        index = self._sample_index(delta, flip)
        if 0 <= index < len(self.kernel):
            return self.kernel[index]
        return 0.0

    def sample_theta_index(self, delta: int, flip: bool = False) -> int:
        """Global theta index driving sample(delta, flip), or -1 if value is fixed / out of range."""
        index = self._sample_index(delta, flip)
        if 0 <= index < len(self.kernel):
            return int(self.global_map[index])
        return -1

    def visualize(self):
        deltas = range(-(self.size//2)-1, self.size//2+2)
        samples = np.array([self.sample(d) for d in deltas])
        return deltas, samples / np.max(samples)


def build_kernel_parameters(start_index, cells, columnar_kernel_size = 7, delta7_kernel_size = 31, symmetric_epgs = False, epg_width = 3, symmetric_delta7 = True):
    parameters = {}
    next_index = start_index
    for cell_type in cells["type"].unique():
        if cell_type == "Delta7":
            input_kernel = KernelParameters(delta7_kernel_size, columnar = False, symmetric = True)
            output_kernel = KernelParameters(columnar_kernel_size, columnar = True, symmetric = symmetric_delta7)
            next_index = input_kernel.allocate_indices(next_index)
            next_index = output_kernel.allocate_indices(next_index)
            parameters[cell_type] = {
                "input": input_kernel,
                "output": output_kernel,
            }
        elif cell_type in {"EPG", "PEG"}:
            # Share input and output kernels
            kernel = KernelParameters(epg_width, columnar = True, symmetric = symmetric_epgs)
            next_index = kernel.allocate_indices(next_index)
            parameters[cell_type] = {
                "input": kernel,
                "output": kernel,
            }
        else:
            # Share input and output kernels
            #kernel = KernelParameters(columnar_kernel_size, columnar = True, symmetric = False)
            input_kernel = KernelParameters(columnar_kernel_size, columnar = True, symmetric = False)
            output_kernel = KernelParameters(columnar_kernel_size, columnar = True, symmetric = False)
            next_index = input_kernel.allocate_indices(next_index)
            next_index = output_kernel.allocate_indices(next_index)
            parameters[cell_type] = {
                "input": input_kernel,
                "output": output_kernel,
            }

    return parameters, next_index

def smooth_max(x, alpha=1.0):
    # For numerical stability, subtract the max value
    max_x = np.max(x)
    
    # Formula: sum(x * exp(alpha * x)) / sum(exp(alpha * x))
    numerator = np.sum(x * np.exp(alpha * (x - max_x)))
    denominator = np.sum(np.exp(alpha * (x - max_x)))
    
    return numerator / denominator


def _smooth_max_value_and_grad_x(x: np.ndarray, alpha: float = 1.0) -> tuple[float, np.ndarray]:
    """Return smooth_max(x) and d(smooth_max)/dx (same weighting as smooth_max)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, np.zeros(0, dtype=np.float64)
    max_x = np.max(x)
    z = np.exp(alpha * (x - max_x))
    z_sum = float(z.sum())
    w = z / z_sum
    y = float(np.dot(w, x))
    grad_x = w * (1.0 + alpha * (x - y))
    return y, grad_x


def _min_overlap_grads(o: float, e: float) -> tuple[float, float]:
    """Gradients of min(o, e) w.r.t. o and e (symmetric subgradient at o == e)."""
    if o < e:
        return 1.0, 0.0
    if o > e:
        return 0.0, 1.0
    return 0.5, 0.5

class Extrapolator:
    def __init__(self, rois, cells, columns_by_roi, projections, kernel_parameters_by_roi, kernel_parameter_count, closed_eb = True):
        self.rois = rois
        self.cells = cells
        self.grouped = cells.group_by("type", "subtype", maintain_order=True).agg(indices = "index").with_columns(count = pl.col("indices").list.len())
        self.types = self.grouped.select(pl.col("type").unique(maintain_order=True)).with_columns(pl.row_index("type_index"))
        self.M = len(self.rois)
        self.N = len(self.types)
        self.columns_by_roi = columns_by_roi
        self.column_index = {}
        for columns in columns_by_roi.values():
            for i, column in enumerate(columns):
                self.column_index[column] = i

        self.kernel_parameters_by_roi = kernel_parameters_by_roi
        self.kernel_parameter_count = kernel_parameter_count
        self.projections = projections
        self.closed_eb = closed_eb

    def _columns_in_sample(self, roi: str, sample: set[str]) -> set[str]:
        return set(self.columns_by_roi[roi]).intersection(sample)

    def _kernel_param_offset(self) -> int:
        return self.M * self.N * self.N

    def _extract_kernels_mapped(self, theta: np.ndarray, roi, cell_type):
        off = self._kernel_param_offset()
        tail = theta[off:]
        kp = self.kernel_parameters_by_roi[roi][cell_type]
        out_k, _ = kp["output"].extract_with_map(tail, index_base=off)
        in_k, _ = kp["input"].extract_with_map(tail, index_base=off)
        return out_k, in_k

    @staticmethod
    def _accum_grad_theta_from_edges(theta_idx_per_k: list, grad_x: np.ndarray) -> dict[int, float]:
        acc: dict[int, float] = {}
        for k, t in enumerate(theta_idx_per_k):
            if t < 0:
                continue
            acc[t] = acc.get(t, 0.0) + float(grad_x[k])
        return acc

    @staticmethod
    def _scale_grad_dict(d: dict[int, float], scale: float) -> dict[int, float]:
        if scale == 0.0:
            return {}
        return {g: scale * v for g, v in d.items()}

    def _build_arborization_derivative_caches(
        self,
        theta: np.ndarray,
        type: str,
        subtype: str,
        roi: str,
        columns: tuple[str, ...],
        alpha: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[int, float]], list[dict[int, float]], str, frozenset]:
        c = len(columns)
        o = np.zeros(c)
        s_vec = np.zeros(c)
        i_mask = np.zeros(c)
        d_o_cols: list[dict[int, float]] = []
        d_s_cols: list[dict[int, float]] = []
        if type not in self.projections[roi] or subtype not in self.projections[roi][type]:
            empty = [{} for _ in range(c)]
            return o, i_mask, s_vec, empty, empty, type, frozenset()
        projections = self.projections[roi][type][subtype]
        proj_set = frozenset(projections)
        out_k, in_k = self._extract_kernels_mapped(theta, roi, type)
        for n, col in enumerate(columns):
            xs_out: list[float] = []
            ts_out: list[int] = []
            for eta in projections:
                d = self.delta(roi, col, eta, type, True)
                fl = self.should_flip(type, eta, subtype)
                xs_out.append(out_k.sample(d, fl))
                ts_out.append(out_k.sample_theta_index(d, fl))
            x_arr = np.asarray(xs_out, dtype=np.float64)
            y_o, gx_o = _smooth_max_value_and_grad_x(x_arr, alpha)
            o[n] = y_o
            d_o_cols.append(self._accum_grad_theta_from_edges(ts_out, gx_o))

            xs_in: list[float] = []
            ts_in: list[int] = []
            for nu in projections:
                d = self.delta(roi, col, nu, type, False)
                fl = self.should_flip(type, nu, subtype)
                xs_in.append(in_k.sample(d, fl))
                ts_in.append(in_k.sample_theta_index(d, fl))
            x_arr_in = np.asarray(xs_in, dtype=np.float64)
            y_s, gx_s = _smooth_max_value_and_grad_x(x_arr_in, alpha)
            s_vec[n] = y_s
            i_mask[n] = 0.0 if type == "Delta7" and (col in projections) else y_s
            d_s_cols.append(self._accum_grad_theta_from_edges(ts_in, gx_s))
        return o, i_mask, s_vec, d_o_cols, d_s_cols, type, proj_set

    def init_theta(self):
        M = self.M
        N = self.N
        return np.hstack([[1] * (M*N*N), np.zeros(self.kernel_parameter_count)])

    def init_random_theta(self):
        M = self.M
        N = self.N
        return np.hstack([[1] * (M*N*N), np.random.random(self.kernel_parameter_count)])

    def theta_len(self):
        M = self.M
        N = self.N
        return M*N*N + self.kernel_parameter_count

    def theta_bounds(self):
        M = self.M
        N = self.N
        lower = np.zeros(self.theta_len())
        upper = np.hstack([np.inf * np.ones(M*N*N), np.ones(self.kernel_parameter_count)])
        return lower, upper

    def extract_scale(self, theta, r, ti, tj):
        M = self.M
        N = self.N
        return theta[r * N * N + ti * N + tj]

    def extract_kernels(self, theta, roi, cell_type) -> tuple[Kernel, Kernel]:
        M = self.M
        N = self.N
        return (self.kernel_parameters_by_roi[roi][cell_type]["output"].extract(theta[M*N*N:]),
                self.kernel_parameters_by_roi[roi][cell_type]["input"].extract(theta[M*N*N:]))

    def mod_delta(self, delta: int, N: int) -> int:
        return (delta + N // 2) % N - N // 2

    def delta(self, roi, a, b, cell_type, output: bool):
        diff = self.column_index[b] - self.column_index[a]
        if self.closed_eb and roi == "EB":
            return self.mod_delta(diff, len(self.columns_by_roi["EB"]))
        else:
            return diff

        # Old
        # diff = np.abs(self.column_index[b] - self.column_index[a])
        # if roi == "EB": # or a_type == "Delta7": # and a_type != "Delta7":
        # #if roi == "EB" or b_type == "Delta7" and a_type != "Delta7":
        #     # since the EB is closed and Delta7 have periodic inputs:
        #     return min(diff, abs(16 - diff))
        # #    #return diff
        # elif roi == "PB" and cell_type == "Delta7" and not output:
        #     return diff
        #     #return min(diff, abs(8 - diff))
        # else:
        #     return diff

    def should_flip(self, type, column, subtype):
        if type == "Delta7":
            return column[0] == "L"
        else:
            return subtype[0] == "L"
        # Hardcoded mirroring logic, should fix
        #return subtype[0] == "L" and (column[0] == "c" or column[0] == "R") or (subtype[0] == "R" and column[0] == "L")
        #return (column[0] == "c" and subtype[0] == "L") or column[0] == "L"
        #return column[0] == "L" or (column[0] == "c" and int(column[1:]) > len(self.columns_by_roi["EB"])//2)

    def build_arborization(self, theta, type, subtype, roi, columns):
        output_arborization = np.zeros(len(columns))
        input_arborization = np.zeros(len(columns))
        self_arborization = np.zeros(len(columns))
        
        # TODO: for all cell types except Delta7, we really expect output and input kernels to be the same.
        # This would probably help a lot with optimization.
        if type not in self.projections[roi] or subtype not in self.projections[roi][type]:
            return output_arborization, input_arborization, self_arborization
        projections = self.projections[roi][type][subtype]

        output_kernel, input_kernel = self.extract_kernels(theta, roi, type)
        for n, col in enumerate(columns):
            output_arborization[n] = smooth_max([output_kernel.sample(self.delta(roi, col, eta, type, True), self.should_flip(type, eta, subtype)) for eta in projections])
            self_arborization[n] = smooth_max([input_kernel.sample(self.delta(roi, col, nu, type, False), self.should_flip(type, nu, subtype)) for nu in projections])
            input_arborization[n] = 0.0 if type == "Delta7" and (col in projections) else self_arborization[n]

        return output_arborization, input_arborization, self_arborization

    def extrapolate(self, theta, sample: set[str]):
        W = np.zeros((len(self.rois), len(self.grouped), len(self.grouped)))
        for r, roi in enumerate(self.rois):
            columns = tuple(sorted(self._columns_in_sample(roi, sample)))
            output_arbs = []
            input_arbs = []
            for type, subtype, indices, count, type_index in self.grouped.join(self.types, on="type").rows():
                output_arb, input_arb, self_arb = self.build_arborization(theta, type, subtype, roi, columns)
                output_arbs.append((type_index, output_arb))
                input_arbs.append((type_index, input_arb, self_arb))

            for i, (ti, output_arb) in enumerate(output_arbs):
                for j, (tj, input_arb, self_arb) in enumerate(input_arbs):
                    # TODO: fix this horrible way of encoding the assumption about Delta7 that they don't take any
                    # input in their output columns except from other Delta7s.
                    if ti == tj:
                        input_arb = self_arb
                    W[r,i,j] += self.extract_scale(theta, r, ti, tj) * np.sum(np.minimum(output_arb, input_arb))

        # No autapses for single-cell groups.
        single = self.grouped["count"].to_numpy() == 1
        W[:, np.arange(len(self.grouped))[single], np.arange(len(self.grouped))[single]] = 0.0

        return W

    def jacobian_extrapolate(
        self, theta: np.ndarray, sample: set[str], alpha: float = 1.0
    ) -> csr_matrix:
        """Jacobian of vec(extrapolate(theta, sample))) w.r.t. theta (row-major W flatten).

        Returns a sparse matrix of shape (M * G * G, theta_len) suitable as ``jac`` for
        ``scipy.optimize.least_squares`` (method ``trf`` or ``dogbox``).
        """
        theta = np.asarray(theta, dtype=np.float64)
        M, N, G = self.M, self.N, len(self.grouped)
        group_counts = self.grouped["count"].to_numpy()
        p = self.theta_len()
        n_rows = M * G * G
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def row_index(r: int, i: int, j: int) -> int:
            return r * G * G + i * G + j

        def scale_index(r: int, ti: int, tj: int) -> int:
            return r * N * N + ti * N + tj

        for r, roi in enumerate(self.rois):
            columns = tuple(sorted(self._columns_in_sample(roi, sample)))
            row_caches: list[dict] = []
            for type, subtype, indices, count, type_index in self.grouped.join(self.types, on="type").rows():
                o, i_mask, s_vec, d_o_cols, d_s_cols, cell_type, proj_set = (
                    self._build_arborization_derivative_caches(
                        theta, type, subtype, roi, columns, alpha=alpha
                    )
                )
                row_caches.append(
                    {
                        "ti": type_index,
                        "type": cell_type,
                        "o": o,
                        "s": s_vec,
                        "i_mask": i_mask,
                        "proj_set": proj_set,
                        "d_o_cols": d_o_cols,
                        "d_s_cols": d_s_cols,
                    }
                )
            for i in range(G):
                for j in range(G):
                    if i == j and group_counts[i] == 1:
                        continue
                    ri = row_index(r, i, j)
                    ti = row_caches[i]["ti"]
                    tj = row_caches[j]["ti"]
                    s_idx = scale_index(r, ti, tj)
                    sval = float(theta[s_idx])
                    o_ar = row_caches[i]["o"]
                    type_j = row_caches[j]["type"]
                    proj_set_j = row_caches[j]["proj_set"]
                    s_ar = row_caches[j]["s"]
                    i_m = row_caches[j]["i_mask"]
                    if ti == tj:
                        eff = s_ar
                    else:
                        eff = i_m
                    rows.append(ri)
                    cols.append(s_idx)
                    data.append(float(np.sum(np.minimum(o_ar, eff))))
                    merged: dict[int, float] = {}
                    for n in range(len(columns)):
                        g_o, g_e = _min_overlap_grads(float(o_ar[n]), float(eff[n]))
                        for g, v in row_caches[i]["d_o_cols"][n].items():
                            merged[g] = merged.get(g, 0.0) + v * g_o
                        if ti == tj:
                            d_eff_n = row_caches[j]["d_s_cols"][n]
                        else:
                            m = 0.0 if type_j == "Delta7" and columns[n] in proj_set_j else 1.0
                            d_eff_n = self._scale_grad_dict(row_caches[j]["d_s_cols"][n], m)
                        for g, v in d_eff_n.items():
                            merged[g] = merged.get(g, 0.0) + v * g_e
                    for g, v in merged.items():
                        rows.append(ri)
                        cols.append(int(g))
                        data.append(sval * v)

        return coo_matrix((data, (rows, cols)), shape=(n_rows, p)).tocsr()

    def group(self, connectome):
        if len(connectome.shape) == 2:
            grouped_connectome = np.zeros((len(self.grouped), len(self.grouped)))
            for i, pre in enumerate(self.grouped["indices"]):
                for j, post in enumerate(self.grouped["indices"]):
                    count = len(pre) * len(post)
                    if np.array_equal(pre, post):
                        count -= len(pre)
                    if count <= 0:
                        continue
                    grouped_connectome[i, j] = np.sum(connectome[np.ix_(pre, post)]) / count
        elif connectome.shape[0] == len(self.rois):
            grouped_connectome = np.zeros((len(self.rois), len(self.grouped), len(self.grouped)))
            for r in range(len(self.rois)):
                for i, pre in enumerate(self.grouped["indices"]):
                    for j, post in enumerate(self.grouped["indices"]):
                        count = len(pre) * len(post)
                        if np.array_equal(pre, post):
                            count -= len(pre)
                        if count <= 0:
                            continue
                        grouped_connectome[r, i, j] = np.sum(connectome[np.ix_([r], pre, post)]) / count

        return grouped_connectome

    def group_var(self, connectome, grouped):
        var = np.zeros((len(self.rois), len(self.grouped), len(self.grouped)))
        for r in range(len(self.rois)):
            for i, pre in enumerate(self.grouped["indices"]):
                for j, post in enumerate(self.grouped["indices"]):
                    count = len(pre) * len(post)
                    if np.array_equal(pre, post):
                        count -= len(pre)
                    if count <= 0:
                        continue
                    var[r, i, j] = np.sum((connectome[np.ix_([r], pre, post)] - grouped[r, i, j])**2) / count
        return var


    def mean_to_sum(self, connectome: np.array):
        connectome = connectome.copy()
        assert connectome.shape[0] == len(self.rois)

        for r in range(len(self.rois)):
            for i, pre in enumerate(self.grouped["indices"]):
                for j, post in enumerate(self.grouped["indices"]):
                    count = len(pre) * len(post)
                    if np.array_equal(pre, post):
                        count -= len(pre)
                    if count <= 0:
                        continue
                    #if count > 1:
                    #    count -= 1
                    connectome[r, i, j] *= count

        return connectome

    def mean_to_mean_input(self, connectome: np.array):
        connectome = connectome.copy()
        assert connectome.shape[0] == len(self.rois)

        #for r in range(len(self.rois)):
        #    for i, pre in enumerate(self.grouped["indices"]):
        #        for j, post in enumerate(self.grouped["indices"]):
        #            count = len(pre)
        #            if np.array_equal(pre, post):
        #                count -= 1
        #            connectome[r, i, j] *= count

        return connectome


if __name__ == "__main__":
    def test_kernel(params: KernelParameters, parameters, expected):
        assert params.allocate_indices(0) == len(parameters)
        cols = []
        samples = []
        kernel = params.extract(np.array(parameters))
        deltas = range(-(params.size//2)-1, params.size//2+2)
        for i in deltas:
            samples.append(kernel.sample(i))
            cols.append(f"{i}:{kernel.sample(i)}")
        print(" ".join(cols), "==", " ".join([f"{i}:{s}" for i, s in zip(deltas, expected)]), "PASS" if np.array_equal(samples, expected) else "FAIL")

    test_kernel(KernelParameters(5, False, False), [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 0])
    test_kernel(KernelParameters(5, False, True), [1, 2, 3], [0, 3, 2, 1, 2, 3, 0])
    test_kernel(KernelParameters(5, True, False), [1, 2, 3, 4], [0, 1, 2, 1, 3, 4, 0])
    test_kernel(KernelParameters(5, True, True), [2, 3], [0, 3, 2, 1, 2, 3, 0])

    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=5)
        y, g = _smooth_max_value_and_grad_x(x, 1.0)
        assert np.isclose(y, smooth_max(x, 1.0), rtol=0, atol=1e-10)
        eps = 1e-7
        fd = np.zeros_like(x)
        for k in range(len(x)):
            e = np.zeros_like(x)
            e[k] = 1.0
            fd[k] = (smooth_max(x + eps * e, 1.0) - smooth_max(x - eps * e, 1.0)) / (2 * eps)
        assert np.allclose(g, fd, rtol=1e-4, atol=1e-4)
    print("smooth_max grad FD check PASS")

    cells = pl.DataFrame({"index": [0], "type": ["T"], "subtype": ["s"]})
    kp, nk = build_kernel_parameters(0, cells)
    ex = Extrapolator(
        rois=["EB"],
        cells=cells,
        columns_by_roi={"EB": ["c1", "c2"]},
        projections={"EB": {"T": {"s": ["c1"]}}},
        kernel_parameters_by_roi={"EB": kp},
        kernel_parameter_count=nk,
    )
    theta = ex.init_theta() + 0.01 * rng.standard_normal(ex.theta_len())
    sample = {"c1"}
    jac = ex.jacobian_extrapolate(theta, sample).toarray()

    def vec_w(th: np.ndarray) -> np.ndarray:
        return ex.extrapolate(th, sample).ravel()

    f0 = vec_w(theta)
    m, p = f0.size, theta.size
    eps = 1e-6
    jac_fd = np.zeros((m, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = 1.0
        jac_fd[:, j] = (vec_w(theta + eps * e) - vec_w(theta - eps * e)) / (2 * eps)
    assert np.allclose(jac, jac_fd, rtol=5e-4, atol=5e-4)
    print("Extrapolator.jacobian_extrapolate FD check PASS")
