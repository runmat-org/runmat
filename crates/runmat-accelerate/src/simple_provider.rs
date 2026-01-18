use crate::host_lu::{lu_factor_host, LuHostFactors};
use crate::sortrows_host::{sort_rows_host, SortRowsHostOutputs};
use anyhow::{anyhow, ensure, Result};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{
    AccelProvider, CorrcoefOptions, CovarianceOptions, FindDirection, FspecialRequest,
    GpuTensorHandle, HostTensorOwned, HostTensorView, ImfilterOptions, PagefunRequest,
    ProviderBandwidth, ProviderCholResult, ProviderCondNorm, ProviderConv1dOptions,
    ProviderConvMode, ProviderConvOrientation, ProviderEigResult, ProviderFindResult,
    ProviderHermitianKind, ProviderIirFilterOptions, ProviderIirFilterResult, ProviderInvOptions,
    ProviderLinsolveOptions, ProviderLinsolveResult, ProviderLuResult, ProviderNanMode,
    ProviderNormOrder, ProviderPinvOptions, ProviderPolyderQuotient, ProviderPrecision,
    ProviderQrOptions, ProviderQrPivot, ProviderQrResult, ProviderScanDirection,
    ProviderSymmetryKind, SetdiffOptions, SetdiffResult, SortComparison, SortResult,
    SortRowsColumnSpec, UniqueOptions, UniqueResult,
};
use runmat_builtins::{Tensor, Value};
use runmat_runtime::RuntimeError;
use runmat_runtime::builtins::array::sorting_sets::unique;
use runmat_runtime::builtins::common::broadcast::{
    broadcast_index as runtime_broadcast_index, broadcast_shapes as runtime_broadcast_shapes,
    compute_strides as runtime_compute_strides,
};
use runmat_runtime::builtins::stats::summary::{
    corrcoef_from_tensors as runtime_corrcoef_from_tensors,
    cov_from_tensors as runtime_cov_from_tensors, CovWeightSpec,
};

use runmat_runtime::builtins::math::linalg::ops::{
    dot_host_real_for_provider, mldivide_host_real_for_provider, mrdivide_host_real_for_provider,
};
use runmat_runtime::builtins::math::linalg::solve::cond::cond_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::inv::inv_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::linsolve::linsolve_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::norm::norm_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::pinv::pinv_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::rank_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::solve::rcond::rcond_host_real_for_provider;
use runmat_runtime::builtins::math::linalg::structure::bandwidth::bandwidth_host_real_data;
use runmat_runtime::builtins::math::linalg::structure::ishermitian::ishermitian_host_real_data;
use runmat_runtime::builtins::math::linalg::structure::issymmetric::issymmetric_host_real_data;
use runmat_runtime::builtins::math::linalg::structure::symrcm::symrcm_host_real_data;
use runmat_runtime::builtins::math::reduction::compute_median_inplace;
use runmat_runtime::builtins::math::reduction::diff_tensor_host;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const PROVIDER_DEFAULT_SEED: u64 = 0x9e3779b97f4a7c15;

static REGISTRY: OnceCell<Mutex<HashMap<u64, Vec<f64>>>> = OnceCell::new();

fn registry() -> &'static Mutex<HashMap<u64, Vec<f64>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

const POLYDER_EPS: f64 = 1.0e-12;
const FACTORIAL_MAX_HOST: usize = 170;
const FACTORIAL_INT_TOL: f64 = 1.0e-10;

fn runtime_flow_to_anyhow(_context: &str, err: RuntimeError) -> anyhow::Error {
    anyhow::Error::new(err)
}

#[derive(Clone, Copy)]
enum PolyOrientation {
    Scalar,
    Row,
    Column,
}

fn poly_orientation_from_shape(shape: &[usize]) -> Result<PolyOrientation> {
    let mut non_unit = 0usize;
    let mut orientation = PolyOrientation::Scalar;
    for (idx, &dim) in shape.iter().enumerate() {
        if dim > 1 {
            non_unit += 1;
            orientation = if idx == 0 {
                PolyOrientation::Column
            } else {
                PolyOrientation::Row
            };
        }
    }
    if non_unit > 1 {
        Err(anyhow!("polyder: coefficient inputs must be vectors"))
    } else {
        Ok(orientation)
    }
}

fn poly_shape_for_len(orientation: PolyOrientation, len: usize) -> Vec<usize> {
    if len <= 1 {
        return vec![1, 1];
    }
    match orientation {
        PolyOrientation::Scalar | PolyOrientation::Row => vec![1, len],
        PolyOrientation::Column => vec![len, 1],
    }
}

fn poly_trim_slice(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    if let Some(idx) = coeffs.iter().position(|c| c.abs() > POLYDER_EPS) {
        coeffs[idx..].to_vec()
    } else {
        vec![0.0]
    }
}

fn poly_raw_derivative(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.len() <= 1 {
        return vec![0.0];
    }
    let mut out = Vec::with_capacity(coeffs.len() - 1);
    let mut power = coeffs.len() - 1;
    for coeff in coeffs.iter().take(coeffs.len() - 1) {
        out.push(*coeff * power as f64);
        power -= 1;
    }
    out
}

fn poly_integral_real(coeffs: &[f64], constant: f64) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![constant];
    }
    let mut out = Vec::with_capacity(coeffs.len() + 1);
    for (idx, &coeff) in coeffs.iter().enumerate() {
        let power = (coeffs.len() - idx) as f64;
        if power == 0.0 {
            out.push(0.0);
        } else {
            out.push(coeff / power);
        }
    }
    out.push(constant);
    out
}

fn poly_convolve_real(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

fn poly_add_real(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    let mut result = vec![0.0; len];
    for (idx, &value) in a.iter().enumerate() {
        result[len - a.len() + idx] += value;
    }
    for (idx, &value) in b.iter().enumerate() {
        result[len - b.len() + idx] += value;
    }
    result
}

fn poly_sub_real(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    let mut result = vec![0.0; len];
    for (idx, &value) in a.iter().enumerate() {
        result[len - a.len() + idx] += value;
    }
    for (idx, &value) in b.iter().enumerate() {
        result[len - b.len() + idx] -= value;
    }
    result
}

fn rng_state() -> &'static Mutex<u64> {
    static RNG: OnceCell<Mutex<u64>> = OnceCell::new();
    RNG.get_or_init(|| Mutex::new(0x9e3779b97f4a7c15))
}

fn factorial_scalar_host(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value == 0.0 {
        return 1.0;
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if value < 0.0 {
        return f64::NAN;
    }
    let rounded = value.round();
    if (value - rounded).abs() > FACTORIAL_INT_TOL {
        return f64::NAN;
    }
    if rounded < 0.0 {
        return f64::NAN;
    }
    let n = rounded as usize;
    if n > FACTORIAL_MAX_HOST {
        return f64::INFINITY;
    }
    if n == 0 {
        return 1.0;
    }
    let mut acc = 1.0f64;
    for k in 2..=n {
        acc *= k as f64;
    }
    acc
}

fn tensor_to_weight_vector(tensor: &Tensor) -> Result<Vec<f64>> {
    if tensor.shape.len() > 2 {
        return Err(anyhow!("covariance: weight vector must be 1-D"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    if rows != 1 && cols != 1 {
        return Err(anyhow!(
            "covariance: weight vector must be 1-D (received shape {}x{})",
            rows,
            cols
        ));
    }
    Ok(tensor.data.clone())
}

fn next_uniform(state: &mut u64) -> f64 {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1;
    const SHIFT: u32 = 11;
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);

    *state = state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
    let bits = *state >> SHIFT;
    (bits as f64) * SCALE
}

fn next_normal_pair(state: &mut u64) -> (f64, f64) {
    let mut u1 = next_uniform(state);
    if u1 <= 0.0 {
        u1 = f64::MIN_POSITIVE;
    }
    let u2 = next_uniform(state);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * std::f64::consts::PI * u2;
    (radius * angle.cos(), radius * angle.sin())
}

pub struct InProcessProvider {
    next_id: AtomicU64,
}

impl InProcessProvider {
    pub const fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
        }
    }

    fn allocate_tensor(&self, data: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, data);
        GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        }
    }

    fn load_polynomial(&self, handle: &GpuTensorHandle) -> Result<(Vec<f64>, PolyOrientation)> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("polyder: unknown tensor handle {}", handle.buffer_id))?
        };
        let orientation = poly_orientation_from_shape(&handle.shape)?;
        let coeffs = if data.is_empty() { vec![0.0] } else { data };
        Ok((coeffs, orientation))
    }

    fn allocate_polynomial(
        &self,
        coeffs: Vec<f64>,
        orientation: PolyOrientation,
    ) -> GpuTensorHandle {
        let shape = poly_shape_for_len(orientation, coeffs.len());
        self.allocate_tensor(coeffs, shape)
    }
}

impl Default for InProcessProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => {
            let n = shape[0];
            vec![n, n]
        }
        _ => shape.to_vec(),
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn product(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn decode_indices(mut index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut coords = Vec::with_capacity(dims.len());
    for &dim in dims {
        if dim == 0 {
            coords.push(0);
        } else {
            let coord = index % dim;
            coords.push(coord);
            index /= dim.max(1);
        }
    }
    coords
}

fn shapes_compatible(expected: &[usize], actual: &[usize]) -> bool {
    let max_len = expected.len().max(actual.len());
    for i in 0..max_len {
        let e = expected.get(i).copied().unwrap_or(1);
        let a = actual.get(i).copied().unwrap_or(1);
        if e != a {
            return false;
        }
    }
    true
}

fn filter_state_shape(mut base: Vec<usize>, dim_idx: usize, state_len: usize) -> Vec<usize> {
    if base.len() <= dim_idx {
        base.extend(std::iter::repeat_n(1, dim_idx + 1 - base.len()));
    }
    if !base.is_empty() {
        base[dim_idx] = state_len;
    }
    base
}

fn states_from_column_major(
    data: &[f64],
    state_len: usize,
    dim_idx: usize,
    shape_ext: &[usize],
) -> Vec<f64> {
    if state_len == 0 {
        return Vec::new();
    }
    let dims_before = &shape_ext[..dim_idx];
    let dims_after = if dim_idx + 1 < shape_ext.len() {
        &shape_ext[dim_idx + 1..]
    } else {
        &[]
    };
    let leading = if dims_before.is_empty() {
        1
    } else {
        dims_before.iter().copied().product()
    };
    let trailing = if dims_after.is_empty() {
        1
    } else {
        dims_after.iter().copied().product()
    };
    let channel_count = leading * trailing;
    let shape = filter_state_shape(shape_ext.to_vec(), dim_idx, state_len);
    let mut states = vec![0.0; state_len * channel_count];
    for channel in 0..channel_count {
        let before_idx = if dims_before.is_empty() {
            0
        } else {
            channel % leading
        };
        let after_idx = if dims_after.is_empty() {
            0
        } else {
            channel / leading
        };
        let before_coords = decode_indices(before_idx, dims_before);
        let after_coords = decode_indices(after_idx, dims_after);
        for s in 0..state_len {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for (d, size) in shape.iter().copied().enumerate() {
                let coord = if d < dim_idx {
                    before_coords.get(d).copied().unwrap_or(0)
                } else if d == dim_idx {
                    s
                } else {
                    let idx = d - dim_idx - 1;
                    after_coords.get(idx).copied().unwrap_or(0)
                };
                offset += coord * stride;
                stride *= size;
            }
            states[channel * state_len + s] = data[offset];
        }
    }
    states
}

fn states_to_column_major(
    states: &[f64],
    state_len: usize,
    dim_idx: usize,
    shape_ext: &[usize],
) -> Vec<f64> {
    if state_len == 0 {
        return Vec::new();
    }
    let dims_before = &shape_ext[..dim_idx];
    let dims_after = if dim_idx + 1 < shape_ext.len() {
        &shape_ext[dim_idx + 1..]
    } else {
        &[]
    };
    let leading = if dims_before.is_empty() {
        1
    } else {
        dims_before.iter().copied().product()
    };
    let trailing = if dims_after.is_empty() {
        1
    } else {
        dims_after.iter().copied().product()
    };
    let channel_count = leading * trailing;
    let shape = filter_state_shape(shape_ext.to_vec(), dim_idx, state_len);
    let mut out = vec![0.0; states.len()];
    for channel in 0..channel_count {
        let before_idx = if dims_before.is_empty() {
            0
        } else {
            channel % leading
        };
        let after_idx = if dims_after.is_empty() {
            0
        } else {
            channel / leading
        };
        let before_coords = decode_indices(before_idx, dims_before);
        let after_coords = decode_indices(after_idx, dims_after);
        for s in 0..state_len {
            let mut offset = 0usize;
            let mut stride = 1usize;
            for (d, size) in shape.iter().copied().enumerate() {
                let coord = if d < dim_idx {
                    before_coords.get(d).copied().unwrap_or(0)
                } else if d == dim_idx {
                    s
                } else {
                    let idx = d - dim_idx - 1;
                    after_coords.get(idx).copied().unwrap_or(0)
                };
                offset += coord * stride;
                stride *= size;
            }
            out[offset] = states[channel * state_len + s];
        }
    }
    out
}

fn permute_data(data: &[f64], shape: &[usize], order: &[usize]) -> Result<(Vec<f64>, Vec<usize>)> {
    ensure!(!order.is_empty(), "permute: order must not be empty");
    let rank = order.len();
    ensure!(
        shape.len() <= rank,
        "permute: order length must be at least the number of dimensions"
    );
    let mut seen = vec![false; rank];
    for &dim in order {
        ensure!(dim < rank, "permute: invalid dimension index {}", dim + 1);
        ensure!(
            !seen[dim],
            "permute: duplicate dimension index {} encountered",
            dim + 1
        );
        seen[dim] = true;
    }
    ensure!(
        seen.iter().all(|v| *v),
        "permute: order must include every dimension exactly once"
    );

    let mut src_shape = shape.to_vec();
    if src_shape.len() < rank {
        src_shape.extend(std::iter::repeat_n(1, rank - src_shape.len()));
    }

    let total = product(&src_shape);
    ensure!(
        total == data.len(),
        "permute: shape/product mismatch ({} vs {})",
        total,
        data.len()
    );

    let mut dst_shape = vec![0usize; rank];
    for (dst_dim, &src_dim) in order.iter().enumerate() {
        dst_shape[dst_dim] = src_shape[src_dim];
    }

    let src_strides = compute_strides(&src_shape);
    let dst_total = product(&dst_shape);
    let mut out = vec![0.0f64; dst_total];
    let mut dst_coords = vec![0usize; rank];
    let mut src_coords = vec![0usize; rank];

    for (dst_index, out_value) in out.iter_mut().enumerate() {
        let mut rem = dst_index;
        for (dim, &size) in dst_shape.iter().enumerate() {
            if size == 0 {
                dst_coords[dim] = 0;
            } else {
                dst_coords[dim] = rem % size;
                rem /= size;
            }
        }
        for (dst_dim, &src_dim) in order.iter().enumerate() {
            src_coords[src_dim] = dst_coords[dst_dim];
        }
        let mut src_index = 0usize;
        for (dim, &coord) in src_coords.iter().enumerate() {
            src_index += coord * src_strides[dim];
        }
        *out_value = data[src_index];
    }

    Ok((out, dst_shape))
}

fn flip_data(data: &[f64], shape: &[usize], axes: &[usize]) -> Result<Vec<f64>> {
    if axes.is_empty() || data.is_empty() {
        return Ok(data.to_vec());
    }
    let mut ext_shape = shape.to_vec();
    if let Some(max_dim) = axes.iter().copied().max() {
        let needed = max_dim + 1;
        if needed > ext_shape.len() {
            ext_shape.extend(std::iter::repeat_n(1, needed - ext_shape.len()));
        }
    }
    let total = product(&ext_shape);
    ensure!(
        total == data.len(),
        "flip: shape/product mismatch ({} vs {})",
        total,
        data.len()
    );
    let mut flip_flags = vec![false; ext_shape.len()];
    for &axis in axes {
        if axis < flip_flags.len() {
            flip_flags[axis] = !flip_flags[axis];
        }
    }
    if !flip_flags.iter().any(|&flag| flag) {
        return Ok(data.to_vec());
    }
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut coords = unravel_index(idx, &ext_shape);
        for (axis, flag) in flip_flags.iter().enumerate() {
            if *flag && ext_shape[axis] > 1 {
                coords[axis] = ext_shape[axis] - 1 - coords[axis];
            }
        }
        let src_idx = ravel_index(&coords, &ext_shape);
        out.push(data[src_idx]);
    }
    Ok(out)
}

fn conv1d_output_shape(len: usize, orientation: ProviderConvOrientation) -> Vec<usize> {
    match (orientation, len) {
        (ProviderConvOrientation::Row, 0) => vec![1, 0],
        (ProviderConvOrientation::Row, _) => vec![1, len],
        (ProviderConvOrientation::Column, 0) => vec![0, 1],
        (ProviderConvOrientation::Column, _) => vec![len, 1],
    }
}

fn convolve_real(signal: &[f64], kernel: &[f64]) -> Result<Vec<f64>> {
    if signal.is_empty() || kernel.is_empty() {
        return Ok(Vec::new());
    }
    let full_len = signal
        .len()
        .checked_add(kernel.len())
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| anyhow!("conv1d: length overflow"))?;
    let mut out = vec![0.0; full_len];
    for (i, &ai) in signal.iter().enumerate() {
        for (j, &bj) in kernel.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    Ok(out)
}

fn apply_conv_mode_real(
    full: &[f64],
    mode: ProviderConvMode,
    len_a: usize,
    len_b: usize,
) -> Vec<f64> {
    match mode {
        ProviderConvMode::Full => full.to_vec(),
        ProviderConvMode::Same => {
            if len_a == 0 {
                return Vec::new();
            }
            let start = if len_b == 0 { 0 } else { (len_b - 1) / 2 };
            let end = (start + len_a).min(full.len());
            if start >= end {
                Vec::new()
            } else {
                full[start..end].to_vec()
            }
        }
        ProviderConvMode::Valid => {
            if len_a < len_b {
                Vec::new()
            } else {
                let start = len_b - 1;
                let valid_len = len_a - len_b + 1;
                let end = (start + valid_len).min(full.len());
                if start >= end {
                    Vec::new()
                } else {
                    full[start..end].to_vec()
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv2d_full_real(
    signal: &[f64],
    signal_rows: usize,
    signal_cols: usize,
    kernel: &[f64],
    kernel_rows: usize,
    kernel_cols: usize,
    full_rows: usize,
    full_cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; full_rows * full_cols];
    for sc in 0..signal_cols {
        for sr in 0..signal_rows {
            let aval = signal[sc * signal_rows + sr];
            if aval == 0.0 {
                continue;
            }
            for kc in 0..kernel_cols {
                let out_c = sc + kc;
                for kr in 0..kernel_rows {
                    let out_r = sr + kr;
                    let kval = kernel[kc * kernel_rows + kr];
                    out[out_c * full_rows + out_r] += aval * kval;
                }
            }
        }
    }
    out
}

fn slice_matrix_real(
    full: &[f64],
    full_rows: usize,
    full_cols: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> (Vec<f64>, usize, usize) {
    let row_end = row_end.min(full_rows);
    let col_end = col_end.min(full_cols);
    if row_start >= row_end || col_start >= col_end {
        let rows = row_end.saturating_sub(row_start);
        let cols = col_end.saturating_sub(col_start);
        return (vec![0.0; rows * cols], rows, cols);
    }
    let rows = row_end - row_start;
    let cols = col_end - col_start;
    let mut out = vec![0.0; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            let src = (col_start + c) * full_rows + (row_start + r);
            let dst = c * rows + r;
            out[dst] = full[src];
        }
    }
    (out, rows, cols)
}

#[allow(clippy::too_many_arguments)]
fn apply_conv2_mode_real_2d(
    full: &[f64],
    full_rows: usize,
    full_cols: usize,
    mode: ProviderConvMode,
    a_rows: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> (Vec<f64>, usize, usize) {
    match mode {
        ProviderConvMode::Full => (full.to_vec(), full_rows, full_cols),
        ProviderConvMode::Same => {
            if a_rows == 0 || a_cols == 0 {
                return (Vec::new(), a_rows, a_cols);
            }
            let row_start = if b_rows == 0 { 0 } else { (b_rows - 1) / 2 };
            let col_start = if b_cols == 0 { 0 } else { (b_cols - 1) / 2 };
            slice_matrix_real(
                full,
                full_rows,
                full_cols,
                row_start,
                row_start + a_rows,
                col_start,
                col_start + a_cols,
            )
        }
        ProviderConvMode::Valid => {
            if a_rows < b_rows || a_cols < b_cols {
                (Vec::new(), 0, 0)
            } else {
                let rows = a_rows - b_rows + 1;
                let cols = a_cols - b_cols + 1;
                let row_start = b_rows.saturating_sub(1);
                let col_start = b_cols.saturating_sub(1);
                slice_matrix_real(
                    full,
                    full_rows,
                    full_cols,
                    row_start,
                    row_start + rows,
                    col_start,
                    col_start + cols,
                )
            }
        }
    }
}

fn tensor_from_value(label: &str, value: Value) -> Result<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| anyhow!("{label}: {e}")),
        Value::Int(i) => {
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| anyhow!("{label}: {e}"))
        }
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| anyhow!("{label}: {e}")),
        Value::ComplexTensor(_) => Err(anyhow!(
            "{label}: complex outputs are not supported by the in-process provider"
        )),
        other => Err(anyhow!("{label}: unexpected value {other:?}")),
    }
}

fn tril_data(data: &[f64], shape: &[usize], offset: isize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let plane = rows.saturating_mul(cols);
    if plane == 0 {
        ensure!(
            data.is_empty(),
            "tril: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(Vec::new());
    }
    let pages = if shape.len() <= 2 {
        1usize
    } else {
        shape[2..].iter().product::<usize>()
    };
    if pages == 0 {
        ensure!(
            data.is_empty(),
            "tril: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(Vec::new());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| anyhow!("tril: dimension product overflow"))?;
    ensure!(
        expected == data.len(),
        "tril: shape/product mismatch ({} vs {})",
        expected,
        data.len()
    );
    let mut out = data.to_vec();
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            for row in 0..rows {
                if (row as isize) - (col as isize) < -offset {
                    out[col_base + row] = 0.0;
                }
            }
        }
    }
    Ok(out)
}

fn triu_data(data: &[f64], shape: &[usize], offset: isize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let plane = rows.saturating_mul(cols);
    if plane == 0 {
        ensure!(
            data.is_empty(),
            "triu: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(Vec::new());
    }
    let pages = if shape.len() <= 2 {
        1usize
    } else {
        shape[2..].iter().product::<usize>()
    };
    if pages == 0 {
        ensure!(
            data.is_empty(),
            "triu: shape/product mismatch ({} vs {})",
            0,
            data.len()
        );
        return Ok(Vec::new());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| anyhow!("triu: dimension product overflow"))?;
    ensure!(
        expected == data.len(),
        "triu: shape/product mismatch ({} vs {})",
        expected,
        data.len()
    );
    let mut out = data.to_vec();
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            for row in 0..rows {
                let diff = (col as isize) - (row as isize);
                if diff < offset {
                    out[col_base + row] = 0.0;
                }
            }
        }
    }
    Ok(out)
}

fn circshift_data(data: &[f64], shape: &[usize], shifts: &[isize]) -> Result<Vec<f64>> {
    ensure!(
        shape.len() == shifts.len(),
        "circshift: shift vector length must match tensor rank"
    );
    let mut total = 1usize;
    for &dim in shape {
        total = total
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("circshift: requested output exceeds maximum size"))?;
    }
    ensure!(
        total == data.len(),
        "circshift: shape/product mismatch ({} vs {})",
        total,
        data.len()
    );
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut normalized = Vec::with_capacity(shape.len());
    for (len, &shift) in shape.iter().zip(shifts.iter()) {
        if *len <= 1 {
            normalized.push(0usize);
            continue;
        }
        let len_isize = *len as isize;
        let mut value = shift % len_isize;
        if value < 0 {
            value += len_isize;
        }
        normalized.push(value as usize);
    }
    if normalized.iter().all(|&s| s == 0) {
        return Ok(data.to_vec());
    }

    let strides = compute_strides(shape);
    let mut out = vec![0.0f64; data.len()];
    for (idx, out_value) in out.iter_mut().enumerate() {
        let coords = unravel_index(idx, shape);
        let mut src_idx = 0usize;
        for (axis, &coord) in coords.iter().enumerate() {
            let len = shape[axis];
            let stride = strides[axis];
            if len <= 1 || normalized[axis] == 0 {
                src_idx += coord * stride;
            } else {
                let shift = normalized[axis] % len;
                let src_coord = (coord + len - shift) % len;
                src_idx += src_coord * stride;
            }
        }
        *out_value = data[src_idx];
    }
    Ok(out)
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            coords.push(0);
        } else {
            coords.push(index % extent);
            index /= extent;
        }
    }
    coords
}

fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for (coord, extent) in coords.iter().zip(shape.iter()) {
        if *extent > 0 {
            index += coord * stride;
            stride *= extent;
        }
    }
    index
}

fn checked_total(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| anyhow!("repmat: requested output exceeds maximum size"))
    })
}

fn repmat_numeric(data: &[f64], shape: &[usize], reps: &[usize]) -> Result<(Vec<f64>, Vec<usize>)> {
    ensure!(
        !reps.is_empty(),
        "repmat: replication factors must be specified"
    );
    let orig_rank = if shape.is_empty() { 1 } else { shape.len() };
    let rank = if reps.len() == 1 {
        orig_rank.max(2)
    } else {
        orig_rank.max(reps.len())
    };

    let mut base_shape = vec![1usize; rank];
    for (idx, &dim) in shape.iter().enumerate() {
        if idx < rank {
            base_shape[idx] = dim;
        }
    }

    let mut factors = vec![1usize; rank];
    if reps.len() == 1 {
        factors.fill(reps[0]);
    } else {
        for (idx, &factor) in reps.iter().enumerate() {
            if idx < rank {
                factors[idx] = factor;
            }
        }
    }

    let mut new_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let scaled = base_shape[i]
            .checked_mul(factors[i])
            .ok_or_else(|| anyhow!("repmat: requested output exceeds maximum size"))?;
        new_shape.push(scaled);
    }

    let orig_total = checked_total(&base_shape)?;
    ensure!(
        orig_total == data.len() || (orig_total == 0 && data.is_empty()),
        "repmat: internal shape mismatch (expected {} elements, found {})",
        orig_total,
        data.len()
    );

    let new_total = checked_total(&new_shape)?;
    if new_total == 0 {
        return Ok((Vec::new(), new_shape));
    }

    let strides = compute_strides(&base_shape);
    let mut out = Vec::with_capacity(new_total);
    for idx in 0..new_total {
        let mut rem = idx;
        let mut src_index = 0usize;
        for dim in 0..rank {
            let dim_size = new_shape[dim];
            let coord = rem % dim_size;
            rem /= dim_size;
            let base = base_shape[dim];
            let orig_coord = if base == 0 { 0 } else { coord % base };
            src_index += orig_coord * strides[dim];
        }
        out.push(data[src_index]);
    }
    Ok((out, new_shape))
}

fn coerce_sub2ind_value(value: f64, dim_number: usize, dim_size: usize) -> Result<usize> {
    if !value.is_finite() {
        return Err(anyhow!(
            "sub2ind: subscript in dimension {} must be finite",
            dim_number
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(anyhow!(
            "sub2ind: subscript in dimension {} must be an integer",
            dim_number
        ));
    }
    if rounded < 1.0 || rounded > dim_size as f64 {
        return Err(anyhow!(
            "sub2ind: subscript {} exceeds dimension {} (size {})",
            rounded as isize,
            dim_number,
            dim_size
        ));
    }
    Ok(rounded as usize)
}

fn identity_data(shape: &[usize]) -> Vec<f64> {
    let shape = normalize_shape(shape);
    let total: usize = shape.iter().copied().product();
    let mut data = vec![0.0; total];
    if shape.is_empty() {
        return data;
    }
    let rows = shape[0];
    let cols = shape[1];
    let diag_len = rows.min(cols);
    if diag_len == 0 {
        return data;
    }
    let strides = compute_strides(&shape);
    let extra_dims = &shape[2..];
    let extra_count = if extra_dims.is_empty() {
        1
    } else {
        extra_dims.iter().copied().product()
    };
    let mut coords = vec![0usize; shape.len()];
    for mut extra_idx in 0..extra_count {
        for (offset, size) in extra_dims.iter().copied().enumerate() {
            let dim = offset + 2;
            if size == 0 {
                coords[dim] = 0;
                continue;
            }
            coords[dim] = extra_idx % size;
            extra_idx /= size;
        }
        for diag in 0..diag_len {
            coords[0] = diag;
            coords[1] = diag;
            let mut linear = 0usize;
            for (dim, &coord) in coords.iter().enumerate() {
                linear += coord * strides[dim];
            }
            data[linear] = 1.0;
        }
    }
    data
}

fn offset_abs(offset: isize) -> usize {
    if offset >= 0 {
        offset as usize
    } else {
        let magnitude = -(offset as i128);
        magnitude as usize
    }
}

fn diag_matrix_size(len: usize, offset: isize) -> Result<(usize, usize)> {
    let shift = offset_abs(offset);
    let size = len
        .checked_add(shift)
        .ok_or_else(|| anyhow!("diag: result dimension exceeds limits"))?;
    let total = size
        .checked_mul(size)
        .ok_or_else(|| anyhow!("diag: result size exceeds limits"))?;
    Ok((size, total))
}

fn diagonal_length(rows: usize, cols: usize, offset: isize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    if offset >= 0 {
        let shift = offset as usize;
        if shift >= cols {
            0
        } else {
            rows.min(cols - shift)
        }
    } else {
        let shift = offset_abs(offset);
        if shift >= rows {
            0
        } else {
            (rows - shift).min(cols)
        }
    }
}

fn diagonal_target_index(idx: usize, offset: isize) -> (usize, usize) {
    if offset >= 0 {
        (idx, idx + offset as usize)
    } else {
        (idx + offset_abs(offset), idx)
    }
}

fn diagonal_source_index(idx: usize, offset: isize) -> (usize, usize) {
    if offset >= 0 {
        (idx, idx + offset as usize)
    } else {
        (idx + offset_abs(offset), idx)
    }
}

fn ensure_diag_shape(label: &str, shape: &[usize]) -> Result<()> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        Err(anyhow!("{label}: input must be 2-D"))
    } else {
        Ok(())
    }
}

fn rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

fn is_vector_like(rows: usize, cols: usize, dims: usize) -> bool {
    rows == 1 || cols == 1 || dims <= 1
}

impl AccelProvider for InProcessProvider {
    fn device_id(&self) -> u32 {
        0
    }
    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap_or_else(|e| e.into_inner());
            guard
                .get(&source.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("gather_linear: unknown buffer {}", source.buffer_id))?
        };
        let mut out = Vec::with_capacity(indices.len());
        for (pos, &idx) in indices.iter().enumerate() {
            let lin = idx as usize;
            ensure!(
                lin < data.len(),
                "gather_linear: index {} (position {}) out of bounds for buffer {} (len={})",
                lin,
                pos,
                source.buffer_id,
                data.len()
            );
            out.push(data[lin]);
        }
        Ok(self.allocate_tensor(out, output_shape.to_vec()))
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        let values_data = {
            let guard = registry().lock().unwrap_or_else(|e| e.into_inner());
            guard.get(&values.buffer_id).cloned().ok_or_else(|| {
                anyhow!("scatter_linear: unknown values buffer {}", values.buffer_id)
            })?
        };
        ensure!(
            values_data.len() == indices.len(),
            "scatter_linear: values length {} does not match indices length {}",
            values_data.len(),
            indices.len()
        );
        let mut guard = registry().lock().unwrap_or_else(|e| e.into_inner());
        let target_buf = guard
            .get_mut(&target.buffer_id)
            .ok_or_else(|| anyhow!("scatter_linear: unknown target buffer {}", target.buffer_id))?;
        for (pos, &idx) in indices.iter().enumerate() {
            let lin = idx as usize;
            ensure!(
                lin < target_buf.len(),
                "scatter_linear: index {} (position {}) out of bounds for target len {}",
                lin,
                pos,
                target_buf.len()
            );
            target_buf[lin] = values_data[pos];
        }
        Ok(())
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = registry().lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(id, host.data.to_vec());
        let handle = GpuTensorHandle {
            shape: host.shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        };
        runmat_accelerate_api::set_handle_logical(&handle, false);
        Ok(handle)
    }

    fn download(&self, h: &GpuTensorHandle) -> Result<HostTensorOwned> {
        let guard = registry().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(buf) = guard.get(&h.buffer_id) {
            Ok(HostTensorOwned {
                data: buf.clone(),
                shape: h.shape.clone(),
            })
        } else {
            Err(anyhow::anyhow!("buffer not found: {}", h.buffer_id))
        }
    }

    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        let mut guard = registry().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&h.buffer_id);
        runmat_accelerate_api::clear_handle_logical(h);
        Ok(())
    }

    fn device_info(&self) -> String {
        "in-process provider (host registry)".to_string()
    }

    fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
        runmat_accelerate_api::ApiDeviceInfo {
            device_id: 0,
            name: "InProcess".to_string(),
            vendor: "RunMat".to_string(),
            memory_bytes: None,
            backend: Some("inprocess".to_string()),
        }
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        runmat_accelerate_api::ProviderTelemetry::default()
    }

    fn reset_telemetry(&self) {}

    fn sort_rows(
        &self,
        handle: &GpuTensorHandle,
        columns: &[SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> Result<SortResult> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("sortrows: unknown buffer {}", handle.buffer_id))?
        };
        let SortRowsHostOutputs {
            values,
            indices,
            indices_shape,
        } = sort_rows_host(&data, &handle.shape, columns, comparison)?;
        Ok(SortResult {
            values: HostTensorOwned {
                data: values,
                shape: handle.shape.clone(),
            },
            indices: HostTensorOwned {
                data: indices,
                shape: indices_shape,
            },
        })
    }

    fn polyder_single(&self, polynomial: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (coeffs, orientation) = self.load_polynomial(polynomial)?;
        let raw = poly_raw_derivative(&coeffs);
        let trimmed = poly_trim_slice(&raw);
        Ok(self.allocate_polynomial(trimmed, orientation))
    }

    fn polyder_product(&self, p: &GpuTensorHandle, q: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (p_coeffs, orientation) = self.load_polynomial(p)?;
        let (q_coeffs, _) = self.load_polynomial(q)?;
        let dp = poly_raw_derivative(&p_coeffs);
        let dq = poly_raw_derivative(&q_coeffs);
        let term1 = poly_convolve_real(&dp, &q_coeffs);
        let term2 = poly_convolve_real(&p_coeffs, &dq);
        let sum = poly_add_real(&term1, &term2);
        let trimmed = poly_trim_slice(&sum);
        Ok(self.allocate_polynomial(trimmed, orientation))
    }

    fn polyder_quotient(
        &self,
        u: &GpuTensorHandle,
        v: &GpuTensorHandle,
    ) -> Result<ProviderPolyderQuotient> {
        let (u_coeffs, orientation_u) = self.load_polynomial(u)?;
        let (v_coeffs, orientation_v) = self.load_polynomial(v)?;
        let du = poly_raw_derivative(&u_coeffs);
        let dv = poly_raw_derivative(&v_coeffs);
        let term1 = poly_convolve_real(&du, &v_coeffs);
        let term2 = poly_convolve_real(&u_coeffs, &dv);
        let numerator_vec = poly_trim_slice(&poly_sub_real(&term1, &term2));
        let denominator_vec = poly_trim_slice(&poly_convolve_real(&v_coeffs, &v_coeffs));
        let numerator = self.allocate_polynomial(numerator_vec, orientation_u);
        let denominator = self.allocate_polynomial(denominator_vec, orientation_v);
        Ok(ProviderPolyderQuotient {
            numerator,
            denominator,
        })
    }

    fn polyint(&self, polynomial: &GpuTensorHandle, constant: f64) -> Result<GpuTensorHandle> {
        let orientation = poly_orientation_from_shape(&polynomial.shape)?;
        let coeffs = {
            let guard = registry().lock().unwrap();
            guard
                .get(&polynomial.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("polyint: unknown tensor handle {}", polynomial.buffer_id))?
        };
        let integrated = poly_integral_real(&coeffs, constant);
        Ok(self.allocate_polynomial(integrated, orientation))
    }

    fn diag_from_vector(&self, vector: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        ensure_diag_shape("diag", &vector.shape)?;
        let (rows, cols) = rows_cols(&vector.shape);
        ensure!(
            is_vector_like(rows, cols, vector.shape.len()),
            "diag: input must be a vector"
        );

        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&vector.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("diag: unknown buffer {}", vector.buffer_id))?
        };
        let len = data.len();
        let (size, total) = diag_matrix_size(len, offset)?;
        let mut out = vec![0.0; total];
        for (idx, &value) in data.iter().enumerate() {
            let (row, col) = diagonal_target_index(idx, offset);
            if row < size && col < size {
                out[row + col * size] = value;
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, out);
        Ok(GpuTensorHandle {
            shape: vec![size, size],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn diag_extract(&self, matrix: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        ensure_diag_shape("diag", &matrix.shape)?;
        let (rows, cols) = rows_cols(&matrix.shape);
        ensure!(
            !is_vector_like(rows, cols, matrix.shape.len()),
            "diag: matrix input required"
        );
        let diag_len = diagonal_length(rows, cols, offset);
        if diag_len == 0 {
            return self.zeros(&[0, 1]);
        }
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("diag: unknown buffer {}", matrix.buffer_id))?
        };
        let mut out = Vec::with_capacity(diag_len);
        for idx in 0..diag_len {
            let (row, col) = diagonal_source_index(idx, offset);
            let linear = row + col * rows;
            out.push(*data.get(linear).unwrap_or(&0.0));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, out);
        Ok(GpuTensorHandle {
            shape: vec![diag_len, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn tril(&self, handle: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("tril: unknown tensor handle {}", handle.buffer_id))?
        };
        let masked = tril_data(&data, &handle.shape, offset)?;
        Ok(self.allocate_tensor(masked, handle.shape.clone()))
    }

    fn triu(&self, handle: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("triu: unknown tensor handle {}", handle.buffer_id))?
        };
        let masked = triu_data(&data, &handle.shape, offset)?;
        Ok(self.allocate_tensor(masked, handle.shape.clone()))
    }

    fn issymmetric(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderSymmetryKind,
        tolerance: f64,
    ) -> Result<bool> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("issymmetric: unknown tensor handle {}", matrix.buffer_id))?
        };
        let skew = matches!(kind, ProviderSymmetryKind::Skew);
        issymmetric_host_real_data(&matrix.shape, &data, skew, tolerance).map_err(|e| anyhow!(e))
    }

    fn ishermitian(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> Result<bool> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("ishermitian: unknown tensor handle {}", matrix.buffer_id))?
        };
        let skew = matches!(kind, ProviderHermitianKind::Skew);
        ishermitian_host_real_data(&matrix.shape, &data, skew, tolerance).map_err(|e| anyhow!(e))
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("bandwidth: unknown tensor handle {}", matrix.buffer_id))?
        };
        let (lower, upper) =
            bandwidth_host_real_data(&matrix.shape, &data).map_err(|e| anyhow!(e))?;
        Ok(ProviderBandwidth {
            lower: lower as u32,
            upper: upper as u32,
        })
    }

    fn sym_rcm(&self, matrix: &GpuTensorHandle) -> Result<Vec<usize>> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("symrcm: unknown tensor handle {}", matrix.buffer_id))?
        };
        symrcm_host_real_data(&matrix.shape, &data).map_err(|e| anyhow!(e))
    }

    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> Result<f64> {
        let guard = registry().lock().unwrap_or_else(|e| e.into_inner());
        let buf = guard
            .get(&h.buffer_id)
            .ok_or_else(|| anyhow!("read_scalar: unknown buffer {}", h.buffer_id))?;
        if linear_index >= buf.len() {
            return Err(anyhow!(
                "read_scalar: index {} out of bounds (len {})",
                linear_index + 1,
                buf.len()
            ));
        }
        Ok(buf[linear_index])
    }

    fn zeros(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = registry().lock().unwrap();
        guard.insert(id, vec![0.0; len]);
        Ok(GpuTensorHandle {
            shape: shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn zeros_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.zeros(&prototype.shape)
    }

    fn ones(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = registry().lock().unwrap();
        guard.insert(id, vec![1.0; len]);
        Ok(GpuTensorHandle {
            shape: shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn ones_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.ones(&prototype.shape)
    }

    fn eye(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let shape = normalize_shape(shape);
        let data = identity_data(&shape);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = registry().lock().unwrap();
        guard.insert(id, data);
        Ok(GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        })
    }

    fn eye_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.eye(&prototype.shape)
    }

    fn linspace(&self, start: f64, stop: f64, count: usize) -> Result<GpuTensorHandle> {
        let data = if count == 0 {
            Vec::new()
        } else if count == 1 {
            vec![stop]
        } else {
            let step = (stop - start) / ((count - 1) as f64);
            let mut seq = Vec::with_capacity(count);
            for idx in 0..count {
                seq.push(start + (idx as f64) * step);
            }
            if let Some(last) = seq.last_mut() {
                *last = stop;
            }
            seq
        };

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, data);
        Ok(GpuTensorHandle {
            shape: vec![1, count],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn random_uniform(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let mut data = vec![0.0; len];
        {
            let mut guard = rng_state().lock().unwrap_or_else(|e| e.into_inner());
            for slot in &mut data {
                *slot = next_uniform(&mut guard);
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut buf_guard = registry().lock().unwrap_or_else(|e| e.into_inner());
        buf_guard.insert(id, data);
        Ok(GpuTensorHandle {
            shape: shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn random_normal(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        let len: usize = shape.iter().copied().product();
        let mut data = Vec::with_capacity(len);
        if len > 0 {
            let mut guard = rng_state().lock().unwrap_or_else(|e| e.into_inner());
            while data.len() < len {
                let (z0, z1) = next_normal_pair(&mut guard);
                data.push(z0);
                if data.len() < len {
                    data.push(z1);
                }
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, data);
        Ok(GpuTensorHandle {
            shape: shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn set_rng_state(&self, state: u64) -> Result<()> {
        let mut guard = rng_state()
            .lock()
            .map_err(|_| anyhow::anyhow!("set_rng_state: RNG mutex poisoned"))?;
        *guard = if state == 0 {
            PROVIDER_DEFAULT_SEED
        } else {
            state
        };
        Ok(())
    }

    fn fspecial(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        let spec =
            runmat_runtime::builtins::image::filters::fspecial::spec_from_request(&request.filter)
                .map_err(|err| anyhow!(err))?;
        let tensor = spec.generate_tensor().map_err(|err| anyhow!(err))?;
        Ok(self.allocate_tensor(tensor.data.clone(), tensor.shape.clone()))
    }

    fn imfilter(
        &self,
        image: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: &ImfilterOptions,
    ) -> Result<GpuTensorHandle> {
        let (image_vec, kernel_vec) = {
            let guard = registry().lock().unwrap();
            let image_buf = guard
                .get(&image.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("imfilter: unknown buffer {}", image.buffer_id))?;
            let kernel_buf = guard
                .get(&kernel.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("imfilter: unknown buffer {}", kernel.buffer_id))?;
            (image_buf, kernel_buf)
        };
        let image_tensor =
            Tensor::new(image_vec, image.shape.clone()).map_err(|e| anyhow!("imfilter: {e}"))?;
        let kernel_tensor =
            Tensor::new(kernel_vec, kernel.shape.clone()).map_err(|e| anyhow!("imfilter: {e}"))?;
        let result = runmat_runtime::builtins::image::filters::imfilter::apply_imfilter_tensor(
            &image_tensor,
            &kernel_tensor,
            options,
            "imfilter",
        )
        .map_err(|err| anyhow!(err))?;
        let Tensor { data, shape, .. } = result;
        Ok(self.allocate_tensor(data, shape))
    }

    fn random_integer_range(
        &self,
        lower: i64,
        upper: i64,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        ensure!(lower <= upper, "lower bound must be <= upper bound");
        let span_i128 = (upper as i128)
            .checked_sub(lower as i128)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| anyhow!("integer range overflow"))?;
        ensure!(span_i128 > 0, "integer range must be non-empty");
        ensure!(
            span_i128 <= (1i128 << 53),
            "integer range exceeds 2^53 and cannot be represented exactly"
        );
        let span = span_i128 as u64;

        let len: usize = shape.iter().copied().product();
        let mut data = Vec::with_capacity(len);
        if span == 1 {
            data.resize(len, lower as f64);
        } else if len > 0 {
            let mut guard = rng_state().lock().unwrap_or_else(|e| e.into_inner());
            let span_f64 = span as f64;
            for _ in 0..len {
                let mut offset = (next_uniform(&mut guard) * span_f64).floor() as u64;
                if offset >= span {
                    offset = span - 1;
                }
                let value = (lower as i128 + offset as i128) as f64;
                data.push(value);
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, data);
        Ok(GpuTensorHandle {
            shape: shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn random_permutation(&self, n: usize, k: usize) -> Result<GpuTensorHandle> {
        ensure!(k <= n, "randperm: K must satisfy 0 <= K <= N");
        let k = k.min(n);
        let mut values: Vec<f64> = if n == 0 {
            Vec::new()
        } else {
            (1..=n).map(|v| v as f64).collect()
        };

        if k > 0 {
            let mut guard = rng_state().lock().unwrap_or_else(|e| e.into_inner());
            for i in 0..k {
                let span = n - i;
                if span == 0 {
                    break;
                }
                let mut u = next_uniform(&mut guard);
                if u >= 1.0 {
                    u = 0.9999999999999999;
                }
                let mut offset = (u * span as f64).floor() as usize;
                if offset >= span {
                    offset = span - 1;
                }
                let j = i + offset;
                values.swap(i, j);
            }
        }

        if values.len() > k {
            values.truncate(k);
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, values);
        Ok(GpuTensorHandle {
            shape: vec![1, k],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn random_permutation_like(
        &self,
        _prototype: &GpuTensorHandle,
        n: usize,
        k: usize,
    ) -> Result<GpuTensorHandle> {
        self.random_permutation(n, k)
    }

    fn covariance(
        &self,
        matrix: &GpuTensorHandle,
        second: Option<&GpuTensorHandle>,
        weights: Option<&GpuTensorHandle>,
        options: &CovarianceOptions,
    ) -> Result<GpuTensorHandle> {
        let host_matrix = self.download(matrix)?;
        let left = Tensor::new(host_matrix.data.clone(), host_matrix.shape.clone())
            .map_err(|e| anyhow!("covariance: {e}"))?;

        let right = if let Some(handle) = second {
            let host = self.download(handle)?;
            Some(
                Tensor::new(host.data.clone(), host.shape.clone())
                    .map_err(|e| anyhow!("covariance: {e}"))?,
            )
        } else {
            None
        };

        let weight_spec = if let Some(handle) = weights {
            let host = self.download(handle)?;
            let tensor = Tensor::new(host.data.clone(), host.shape.clone())
                .map_err(|e| anyhow!("covariance: {e}"))?;
            let vec = tensor_to_weight_vector(&tensor).map_err(|e| anyhow!("covariance: {e}"))?;
            CovWeightSpec::Vector(vec)
        } else {
            CovWeightSpec::Scalar(options.normalization)
        };

        let result = runtime_cov_from_tensors(left, right, options.rows, weight_spec)
            .map_err(|flow| runtime_flow_to_anyhow("covariance", flow))?;

        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        self.upload(&view)
    }

    fn corrcoef(
        &self,
        matrix: &GpuTensorHandle,
        options: &CorrcoefOptions,
    ) -> Result<GpuTensorHandle> {
        let host = self.download(matrix)?;
        let tensor = Tensor::new(host.data.clone(), host.shape.clone())
            .map_err(|e| anyhow!("corrcoef: {e}"))?;
        let result = runtime_corrcoef_from_tensors(tensor, None, options.normalization, options.rows)
            .map_err(|flow| runtime_flow_to_anyhow("corrcoef", flow))?;
        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        self.upload(&view)
    }

    fn elem_add(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i] + bbuf[i];
        }
        drop(guard);
        // Upload new buffer to registry
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_mul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i] * bbuf[i];
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_sub(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i] - bbuf[i];
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_div(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = if bbuf[i] == 0.0 {
                f64::INFINITY * abuf[i].signum()
            } else {
                abuf[i] / bbuf[i]
            };
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_pow(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i].powf(bbuf[i]);
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_ne(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("elem_ne: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("elem_ne: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape = runtime_broadcast_shapes("ne", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs != rhs { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn elem_ge(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("elem_ge: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("elem_ge: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape = runtime_broadcast_shapes("ge", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs >= rhs { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn elem_le(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("elem_le: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("elem_le: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape = runtime_broadcast_shapes("le", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs <= rhs { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn elem_lt(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("elem_lt: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("elem_lt: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape = runtime_broadcast_shapes("lt", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs < rhs { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn elem_gt(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("elem_gt: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("elem_gt: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape = runtime_broadcast_shapes("gt", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs > rhs { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn elem_eq(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("elem_eq: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("elem_eq: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape = runtime_broadcast_shapes("eq", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs == rhs { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_and(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_and: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("logical_and: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape =
            runtime_broadcast_shapes("logical_and", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs != 0.0 && rhs != 0.0 { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_or(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_or: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("logical_or: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape =
            runtime_broadcast_shapes("logical_or", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            out.push(if lhs != 0.0 || rhs != 0.0 { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_xor(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (adata, bdata) = {
            let guard = registry().lock().unwrap();
            let a_vec = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_xor: unknown buffer {}", a.buffer_id))?
                .clone();
            let b_vec = guard
                .get(&b.buffer_id)
                .ok_or_else(|| anyhow!("logical_xor: unknown buffer {}", b.buffer_id))?
                .clone();
            (a_vec, b_vec)
        };

        let shape =
            runtime_broadcast_shapes("logical_xor", &a.shape, &b.shape).map_err(|e| anyhow!(e))?;
        let total: usize = shape.iter().copied().product();
        if total == 0 {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let strides_a = runtime_compute_strides(&a.shape);
        let strides_b = runtime_compute_strides(&b.shape);
        let len_a = adata.len();
        let len_b = bdata.len();

        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            let lhs = if len_a == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &a.shape, &strides_a);
                *adata.get(offset).unwrap_or(&0.0)
            };
            let rhs = if len_b == 0 {
                0.0
            } else {
                let offset = runtime_broadcast_index(idx, &shape, &b.shape, &strides_b);
                *bdata.get(offset).unwrap_or(&0.0)
            };
            let lhs_true = lhs != 0.0;
            let rhs_true = rhs != 0.0;
            out.push(if lhs_true ^ rhs_true { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_not(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_not: unknown buffer {}", a.buffer_id))?
                .clone()
        };

        let shape = a.shape.clone();
        if data.is_empty() {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let mut out = Vec::with_capacity(data.len());
        for value in data {
            out.push(if value == 0.0 { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_isreal(&self, a: &GpuTensorHandle) -> Result<bool> {
        {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_isreal: unknown buffer {}", a.buffer_id))?;
        }
        Ok(true)
    }

    fn logical_isfinite(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_isfinite: unknown buffer {}", a.buffer_id))?
                .clone()
        };

        let shape = a.shape.clone();
        if data.is_empty() {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let mut out = Vec::with_capacity(data.len());
        for value in data {
            out.push(if value.is_finite() { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_isnan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_isnan: unknown buffer {}", a.buffer_id))?
                .clone()
        };

        let shape = a.shape.clone();
        if data.is_empty() {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let mut out = Vec::with_capacity(data.len());
        for value in data {
            out.push(if value.is_nan() { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn logical_isinf(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow!("logical_isinf: unknown buffer {}", a.buffer_id))?
                .clone()
        };

        let shape = a.shape.clone();
        if data.is_empty() {
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let mut out = Vec::with_capacity(data.len());
        for value in data {
            out.push(if value.is_infinite() { 1.0 } else { 0.0 });
        }

        Ok(self.allocate_tensor(out, shape))
    }

    fn elem_hypot(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i].hypot(bbuf[i]);
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn elem_atan2(&self, y: &GpuTensorHandle, x: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let ybuf = guard
            .get(&y.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", y.buffer_id))?;
        let xbuf = guard
            .get(&x.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", x.buffer_id))?;
        if y.shape != x.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; ybuf.len()];
        for idx in 0..ybuf.len() {
            out[idx] = ybuf[idx].atan2(xbuf[idx]);
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: y.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_sin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.sin()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_gamma(&self, _a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_gamma not supported by provider"))
    }
    fn unary_factorial(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().copied().map(factorial_scalar_host).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_asinh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.asinh()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_sinh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.sinh()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_cosh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.cosh()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_asin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.asin()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_acos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.acos()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_acosh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.acosh()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_tan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.tan()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_tanh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.tanh()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_atan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.atan()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }
    fn unary_atanh(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.atanh()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_ceil(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.ceil()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_floor(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.floor()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_round(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.round()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_fix(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf
            .iter()
            .map(|&x| {
                if !x.is_finite() {
                    x
                } else {
                    let truncated = x.trunc();
                    if truncated == 0.0 {
                        0.0
                    } else {
                        truncated
                    }
                }
            })
            .collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_cos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.cos()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_abs(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.abs()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_exp(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.exp()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_log(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.ln()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_sqrt(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.sqrt()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_double(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?
            .clone();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, abuf);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_single(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| (x as f32) as f64).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn unary_pow2(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.exp2()).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn pow2_scale(
        &self,
        mantissa: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let mbuf = guard
            .get(&mantissa.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", mantissa.buffer_id))?;
        let ebuf = guard
            .get(&exponent.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", exponent.buffer_id))?;
        if mantissa.shape != exponent.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0f64; mbuf.len()];
        for i in 0..mbuf.len() {
            out[i] = mbuf[i] * ebuf[i].exp2();
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: mantissa.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn scalar_add(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x + scalar).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn scalar_sub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x - scalar).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x * scalar).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn scalar_div(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = if scalar == 0.0 {
            abuf.iter().map(|&x| f64::INFINITY * x.signum()).collect()
        } else {
            abuf.iter().map(|&x| x / scalar).collect()
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn scalar_rsub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        // compute scalar - a
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| scalar - x).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn scalar_rdiv(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        // compute scalar ./ a
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf
            .iter()
            .map(|&x| {
                if x == 0.0 {
                    f64::INFINITY * scalar.signum()
                } else {
                    scalar / x
                }
            })
            .collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn transpose(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("transpose: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let mut out = vec![0.0; abuf.len()];
        for i in 0..rows {
            for j in 0..cols {
                let src = i + j * rows;
                let dst = j + i * cols;
                out[dst] = abuf[src];
            }
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: vec![cols, rows],
            device_id: 0,
            buffer_id: id,
        })
    }
    fn conv1d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        let signal_len: usize = signal.shape.iter().copied().product();
        let kernel_len: usize = kernel.shape.iter().copied().product();

        if signal_len == 0 || kernel_len == 0 {
            let shape = conv1d_output_shape(0, options.orientation);
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        if matches!(options.mode, ProviderConvMode::Valid) && signal_len < kernel_len {
            let shape = conv1d_output_shape(0, options.orientation);
            return Ok(self.allocate_tensor(Vec::new(), shape));
        }

        let (signal_data, kernel_data) = {
            let guard = registry().lock().unwrap();
            let signal_buf = guard
                .get(&signal.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("conv1d: unknown signal buffer {}", signal.buffer_id))?;
            let kernel_buf = guard
                .get(&kernel.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("conv1d: unknown kernel buffer {}", kernel.buffer_id))?;
            (signal_buf, kernel_buf)
        };

        ensure!(
            signal_data.len() == signal_len,
            "conv1d: signal length mismatch (shape implies {}, buffer has {})",
            signal_len,
            signal_data.len()
        );
        ensure!(
            kernel_data.len() == kernel_len,
            "conv1d: kernel length mismatch (shape implies {}, buffer has {})",
            kernel_len,
            kernel_data.len()
        );

        let full = convolve_real(&signal_data, &kernel_data)?;
        let shaped = apply_conv_mode_real(&full, options.mode, signal_len, kernel_len);
        let shape = conv1d_output_shape(shaped.len(), options.orientation);
        Ok(self.allocate_tensor(shaped, shape))
    }
    fn conv2d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        ensure_diag_shape("conv2d", &signal.shape)?;
        ensure_diag_shape("conv2d", &kernel.shape)?;
        let (signal_rows, signal_cols) = rows_cols(&signal.shape);
        let (kernel_rows, kernel_cols) = rows_cols(&kernel.shape);

        let signal_len: usize = signal.shape.iter().copied().product();
        let kernel_len: usize = kernel.shape.iter().copied().product();

        let empty_shape = match mode {
            ProviderConvMode::Full | ProviderConvMode::Valid => vec![0, 0],
            ProviderConvMode::Same => vec![signal_rows, signal_cols],
        };

        if signal_len == 0
            || kernel_len == 0
            || signal_rows == 0
            || signal_cols == 0
            || kernel_rows == 0
            || kernel_cols == 0
        {
            let len = empty_shape.iter().copied().product::<usize>();
            return Ok(self.allocate_tensor(vec![0.0; len], empty_shape));
        }

        let (signal_data, kernel_data) = {
            let guard = registry().lock().unwrap();
            let signal_buf = guard
                .get(&signal.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("conv2d: unknown signal buffer {}", signal.buffer_id))?;
            let kernel_buf = guard
                .get(&kernel.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("conv2d: unknown kernel buffer {}", kernel.buffer_id))?;
            (signal_buf, kernel_buf)
        };

        ensure!(
            signal_data.len() == signal_len,
            "conv2d: signal length mismatch (shape implies {}, buffer has {})",
            signal_len,
            signal_data.len()
        );
        ensure!(
            kernel_data.len() == kernel_len,
            "conv2d: kernel length mismatch (shape implies {}, buffer has {})",
            kernel_len,
            kernel_data.len()
        );

        let full_rows = signal_rows
            .checked_add(kernel_rows)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| anyhow!("conv2d: row size overflow"))?;
        let full_cols = signal_cols
            .checked_add(kernel_cols)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| anyhow!("conv2d: column size overflow"))?;
        full_rows
            .checked_mul(full_cols)
            .ok_or_else(|| anyhow!("conv2d: output size overflow"))?;

        let full = conv2d_full_real(
            &signal_data,
            signal_rows,
            signal_cols,
            &kernel_data,
            kernel_rows,
            kernel_cols,
            full_rows,
            full_cols,
        );

        let (shaped, out_rows, out_cols) = apply_conv2_mode_real_2d(
            &full,
            full_rows,
            full_cols,
            mode,
            signal_rows,
            signal_cols,
            kernel_rows,
            kernel_cols,
        );
        Ok(self.allocate_tensor(shaped, vec![out_rows, out_cols]))
    }
    fn iir_filter(
        &self,
        b: &GpuTensorHandle,
        a: &GpuTensorHandle,
        x: &GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> Result<ProviderIirFilterResult> {
        let ProviderIirFilterOptions { dim, zi } = options;

        let nb = product(&b.shape);
        let na = product(&a.shape);
        ensure!(
            nb > 0,
            "iir_filter: numerator coefficients must not be empty"
        );
        ensure!(
            na > 0,
            "iir_filter: denominator coefficients must not be empty"
        );

        let signal_elems = product(&x.shape);
        let zi_shape = zi.as_ref().map(|handle| handle.shape.clone());

        let (b_data, a_data, x_data, zi_data) =
            {
                let guard = registry().lock().unwrap();
                let b_buf = guard.get(&b.buffer_id).cloned().ok_or_else(|| {
                    anyhow!("iir_filter: unknown numerator buffer {}", b.buffer_id)
                })?;
                let a_buf = guard.get(&a.buffer_id).cloned().ok_or_else(|| {
                    anyhow!("iir_filter: unknown denominator buffer {}", a.buffer_id)
                })?;
                let x_buf = guard
                    .get(&x.buffer_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("iir_filter: unknown signal buffer {}", x.buffer_id))?;
                ensure!(
                    b_buf.len() == nb,
                    "iir_filter: numerator length mismatch (shape implies {}, buffer has {})",
                    nb,
                    b_buf.len()
                );
                ensure!(
                    a_buf.len() == na,
                    "iir_filter: denominator length mismatch (shape implies {}, buffer has {})",
                    na,
                    a_buf.len()
                );
                ensure!(
                    x_buf.len() == signal_elems,
                    "iir_filter: signal length mismatch (shape implies {}, buffer has {})",
                    signal_elems,
                    x_buf.len()
                );
                let zi_buf = if let Some(ref zi_handle) = zi {
                    Some(guard.get(&zi_handle.buffer_id).cloned().ok_or_else(|| {
                        anyhow!(
                            "iir_filter: unknown initial state buffer {}",
                            zi_handle.buffer_id
                        )
                    })?)
                } else {
                    None
                };
                (b_buf, a_buf, x_buf, zi_buf)
            };

        ensure!(
            !a_data.is_empty() && a_data[0] != 0.0,
            "iir_filter: denominator coefficient a(1) must be non-zero"
        );

        let mut shape_ext = x.shape.clone();
        if dim >= shape_ext.len() {
            shape_ext.extend(std::iter::repeat_n(1, dim + 1 - shape_ext.len()));
        }
        let dim_idx = dim;
        let dim_len = shape_ext.get(dim_idx).copied().unwrap_or(1);

        let leading = if dim_idx == 0 {
            1
        } else {
            shape_ext[..dim_idx].iter().copied().product()
        };
        let trailing = if dim_idx + 1 >= shape_ext.len() {
            1
        } else {
            shape_ext[dim_idx + 1..].iter().copied().product()
        };
        let channel_count = leading * trailing;

        let order = nb.max(na);
        let state_len = order.saturating_sub(1);
        let state_shape = filter_state_shape(shape_ext.clone(), dim_idx, state_len);
        let expected_states = state_len.saturating_mul(channel_count);

        if let Some(ref shape) = zi_shape {
            ensure!(
                shapes_compatible(&state_shape, shape),
                "iir_filter: initial conditions are not compatible with the signal shape"
            );
            let zi_dim = if dim_idx < shape.len() {
                shape[dim_idx]
            } else {
                1
            };
            ensure!(
                zi_dim == state_len,
                "iir_filter: initial conditions must have {} states along dimension {}",
                state_len,
                dim + 1
            );
        }

        let mut states = if state_len == 0 {
            Vec::new()
        } else if let Some(ref zi_buf) = zi_data {
            ensure!(
                zi_buf.len() == expected_states,
                "iir_filter: initial state vector length mismatch (expected {}, found {})",
                expected_states,
                zi_buf.len()
            );
            states_from_column_major(zi_buf, state_len, dim_idx, &shape_ext)
        } else {
            vec![0.0; expected_states]
        };

        let mut b_norm = vec![0.0f64; order];
        let mut a_norm = vec![0.0f64; order];
        let a0 = a_data[0];
        for i in 0..order {
            let b_coeff = if i < nb { b_data[i] } else { 0.0 };
            b_norm[i] = b_coeff / a0;
            if i == 0 {
                a_norm[0] = 1.0;
            } else {
                let a_coeff = if i < na { a_data[i] } else { 0.0 };
                a_norm[i] = a_coeff / a0;
            }
        }

        let mut output = vec![0.0f64; x_data.len()];

        if state_len == 0 {
            let gain = b_norm[0];
            for (dst, &src) in output.iter_mut().zip(x_data.iter()) {
                *dst = gain * src;
            }
        } else if dim_len == 0 || channel_count == 0 {
            // No samples to process; states remain unchanged.
        } else {
            for t in 0..trailing {
                let base = t
                    .checked_mul(dim_len)
                    .and_then(|v| v.checked_mul(leading))
                    .ok_or_else(|| anyhow!("iir_filter: index overflow"))?;
                for l in 0..leading {
                    let channel_idx = t
                        .checked_mul(leading)
                        .and_then(|v| v.checked_add(l))
                        .ok_or_else(|| anyhow!("iir_filter: index overflow"))?;
                    if channel_idx >= channel_count {
                        continue;
                    }
                    let state_base = channel_idx
                        .checked_mul(state_len)
                        .ok_or_else(|| anyhow!("iir_filter: state index overflow"))?;
                    for step in 0..dim_len {
                        let idx = base
                            .checked_add(l)
                            .and_then(|v| v.checked_add(step.saturating_mul(leading)))
                            .ok_or_else(|| anyhow!("iir_filter: signal index overflow"))?;
                        if idx >= x_data.len() {
                            break;
                        }
                        let x_n = x_data[idx];
                        let y = b_norm[0] * x_n + states.get(state_base).copied().unwrap_or(0.0);
                        output[idx] = y;
                        for i in 1..order {
                            let next_state = if i < state_len {
                                states[state_base + i]
                            } else {
                                0.0
                            };
                            let new_state = b_norm[i] * x_n + next_state - a_norm[i] * y;
                            states[state_base + i - 1] = new_state;
                        }
                    }
                }
            }
        }

        let final_state_data = if state_len == 0 {
            Vec::new()
        } else {
            states_to_column_major(&states, state_len, dim_idx, &shape_ext)
        };

        let output_handle = self.allocate_tensor(output, x.shape.clone());
        let final_state_handle = self.allocate_tensor(final_state_data, state_shape);

        Ok(ProviderIirFilterResult {
            output: output_handle,
            final_state: Some(final_state_handle),
        })
    }
    fn permute(&self, handle: &GpuTensorHandle, order: &[usize]) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow!("permute: unknown tensor handle {}", handle.buffer_id))?
                .clone()
        };
        let (permuted, new_shape) = permute_data(&data, &handle.shape, order)?;
        Ok(self.allocate_tensor(permuted, new_shape))
    }

    fn flip(&self, handle: &GpuTensorHandle, axes: &[usize]) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow!("flip: unknown tensor handle {}", handle.buffer_id))?
                .clone()
        };
        let flipped = flip_data(&data, &handle.shape, axes)?;
        Ok(self.allocate_tensor(flipped, handle.shape.clone()))
    }

    fn circshift(&self, handle: &GpuTensorHandle, shifts: &[isize]) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow!("circshift: unknown tensor handle {}", handle.buffer_id))?
                .clone()
        };
        let mut shape = handle.shape.clone();
        if shifts.len() > shape.len() {
            shape.extend(std::iter::repeat_n(1, shifts.len() - shape.len()));
        }
        let mut full_shifts = vec![0isize; shape.len()];
        for (idx, &shift) in shifts.iter().enumerate() {
            if idx < full_shifts.len() {
                full_shifts[idx] = shift;
            }
        }
        let rotated = circshift_data(&data, &shape, &full_shifts)?;
        Ok(self.allocate_tensor(rotated, shape))
    }

    fn diff_dim(
        &self,
        handle: &GpuTensorHandle,
        order: usize,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        if order == 0 {
            return Ok(handle.clone());
        }
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow!("diff_dim: unknown tensor handle {}", handle.buffer_id))?
                .clone()
        };
        let tensor =
            Tensor::new(data, handle.shape.clone()).map_err(|e| anyhow!("diff_dim: {e}"))?;
        let diffed =
            diff_tensor_host(tensor, order, Some(dim + 1)).map_err(|e| anyhow!("diff_dim: {e}"))?;
        let Tensor { data, shape, .. } = diffed;
        Ok(self.allocate_tensor(data, shape))
    }

    fn unique(&self, handle: &GpuTensorHandle, options: &UniqueOptions) -> Result<UniqueResult> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow!("unique: unknown tensor handle {}", handle.buffer_id))?
                .clone()
        };
        let tensor = Tensor::new(data, handle.shape.clone()).map_err(|e| anyhow!("unique: {e}"))?;
        let eval = match unique::unique_numeric_from_tensor(tensor, options) {
            Ok(eval) => eval,
            Err(err) => {
                return Err(anyhow!("{err}"));
            }
        };
        match eval.into_numeric_unique_result() {
            Ok(result) => Ok(result),
            Err(err) => Err(anyhow!("{err}")),
        }
    }

    fn setdiff(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        options: &SetdiffOptions,
    ) -> Result<SetdiffResult> {
        let data_a = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("setdiff: unknown tensor handle {}", a.buffer_id))?
        };
        let data_b = {
            let guard = registry().lock().unwrap();
            guard
                .get(&b.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("setdiff: unknown tensor handle {}", b.buffer_id))?
        };
        let tensor_a = Tensor::new(data_a, a.shape.clone()).map_err(|e| anyhow!("setdiff: {e}"))?;
        let tensor_b = Tensor::new(data_b, b.shape.clone()).map_err(|e| anyhow!("setdiff: {e}"))?;
        let eval = match runmat_runtime::builtins::array::sorting_sets::setdiff::setdiff_numeric_from_tensors(
            tensor_a, tensor_b, options,
        ) {
            Ok(eval) => eval,
            Err(err) => {
                return Err(anyhow!("setdiff: {err}"));
            }
        };
        match eval.into_numeric_setdiff_result() {
            Ok(result) => Ok(result),
            Err(err) => Err(anyhow!("setdiff: {err}")),
        }
    }

    fn repmat(&self, handle: &GpuTensorHandle, reps: &[usize]) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow!("repmat: unknown tensor handle {}", handle.buffer_id))?
                .clone()
        };
        let (tiled, shape) = repmat_numeric(&data, &handle.shape, reps)?;
        Ok(self.allocate_tensor(tiled, shape))
    }

    fn dot(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let (lhs_buf, rhs_buf) = {
            let guard = registry().lock().unwrap();
            let lhs_buf = guard
                .get(&lhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("dot: unknown tensor handle {}", lhs.buffer_id))?;
            let rhs_buf = guard
                .get(&rhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("dot: unknown tensor handle {}", rhs.buffer_id))?;
            (lhs_buf, rhs_buf)
        };
        let lhs_tensor =
            Tensor::new(lhs_buf, lhs.shape.clone()).map_err(|e| anyhow!("dot: {e}"))?;
        let rhs_tensor =
            Tensor::new(rhs_buf, rhs.shape.clone()).map_err(|e| anyhow!("dot: {e}"))?;
        let result =
            dot_host_real_for_provider(&lhs_tensor, &rhs_tensor, dim).map_err(|e| anyhow!(e))?;
        Ok(self.allocate_tensor(result.data.clone(), result.shape.clone()))
    }

    fn reduce_sum(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let s: f64 = abuf.iter().sum();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, vec![s]);
        Ok(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_sum_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("reduce_sum_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let (out, shape) = match dim {
            0 => {
                // sum over rows -> 1 x cols
                let mut v = vec![0.0f64; cols];
                for c in 0..cols {
                    let mut s = 0.0;
                    for r in 0..rows {
                        s += abuf[r + c * rows];
                    }
                    v[c] = s;
                }
                (v, vec![1, cols])
            }
            1 => {
                // sum over cols -> rows x 1
                let mut v = vec![0.0f64; rows];
                for r in 0..rows {
                    let mut s = 0.0;
                    for c in 0..cols {
                        s += abuf[r + c * rows];
                    }
                    v[r] = s;
                }
                (v, vec![rows, 1])
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "reduce_sum_dim: only dims 0 or 1 supported"
                ))
            }
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_prod(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let p: f64 = abuf.iter().product();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, vec![p]);
        Ok(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_prod_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("reduce_prod_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out = if dim <= 1 {
            let mut v = vec![1.0f64; cols];
            for c in 0..cols {
                let mut prod = 1.0;
                for r in 0..rows {
                    prod *= abuf[r + c * rows];
                }
                v[c] = prod;
            }
            v
        } else {
            let mut v = vec![1.0f64; rows];
            for r in 0..rows {
                let mut prod = 1.0;
                for c in 0..cols {
                    prod *= abuf[r + c * rows];
                }
                v[r] = prod;
            }
            v
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        let shape = if dim <= 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        Ok(GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_mean(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let mean = if abuf.is_empty() {
            0.0
        } else {
            abuf.iter().sum::<f64>() / (abuf.len() as f64)
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![mean]);
        Ok(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_mean_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("reduce_mean_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out = if dim <= 1 {
            let mut v = vec![0.0f64; cols];
            for c in 0..cols {
                let mut s = 0.0;
                for r in 0..rows {
                    s += abuf[r + c * rows];
                }
                v[c] = s / (rows as f64);
            }
            v
        } else {
            let mut v = vec![0.0f64; rows];
            for r in 0..rows {
                let mut s = 0.0;
                for c in 0..cols {
                    s += abuf[r + c * rows];
                }
                v[r] = s / (cols as f64);
            }
            v
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, out);
        let shape = if dim <= 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        Ok(GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_any(&self, a: &GpuTensorHandle, omit_nan: bool) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("reduce_any: unknown tensor handle {}", a.buffer_id))?
        };
        let mut truthy = false;
        for v in &data {
            if v.is_nan() {
                if !omit_nan {
                    truthy = true;
                    break;
                }
            } else if *v != 0.0 {
                truthy = true;
                break;
            }
        }
        let result = if truthy { 1.0 } else { 0.0 };
        Ok(self.allocate_tensor(vec![result], vec![1, 1]))
    }

    fn reduce_any_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow!("reduce_any_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("reduce_any_dim: unknown tensor handle {}", a.buffer_id))?
        };
        let (out, shape) = if dim == 0 {
            let mut v = vec![0.0f64; cols];
            for c in 0..cols {
                let mut truth = false;
                for r in 0..rows {
                    let val = data[r + c * rows];
                    if val.is_nan() {
                        if !omit_nan {
                            truth = true;
                            break;
                        }
                    } else if val != 0.0 {
                        truth = true;
                        break;
                    }
                }
                v[c] = if truth { 1.0 } else { 0.0 };
            }
            (v, vec![1, cols])
        } else if dim == 1 {
            let mut v = vec![0.0f64; rows];
            for r in 0..rows {
                let mut truth = false;
                for c in 0..cols {
                    let val = data[r + c * rows];
                    if val.is_nan() {
                        if !omit_nan {
                            truth = true;
                            break;
                        }
                    } else if val != 0.0 {
                        truth = true;
                        break;
                    }
                }
                v[r] = if truth { 1.0 } else { 0.0 };
            }
            (v, vec![rows, 1])
        } else {
            return Err(anyhow!("reduce_any_dim: invalid dimension {}", dim));
        };
        Ok(self.allocate_tensor(out, shape))
    }

    fn reduce_all(&self, a: &GpuTensorHandle, omit_nan: bool) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("reduce_all: unknown tensor handle {}", a.buffer_id))?
        };
        let mut all_true = true;
        let mut saw_value = false;
        for v in &data {
            if v.is_nan() {
                if omit_nan {
                    continue;
                }
            } else {
                saw_value = true;
                if *v == 0.0 {
                    all_true = false;
                    break;
                }
                continue;
            }
        }
        if omit_nan && !saw_value {
            all_true = true;
        }
        let result = if all_true { 1.0 } else { 0.0 };
        Ok(self.allocate_tensor(vec![result], vec![1, 1]))
    }

    fn reduce_all_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow!("reduce_all_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("reduce_all_dim: unknown tensor handle {}", a.buffer_id))?
        };
        let (out, shape) = if dim == 0 {
            let mut v = vec![0.0f64; cols];
            for c in 0..cols {
                let mut all_true = true;
                let mut saw_value = false;
                for r in 0..rows {
                    let val = data[r + c * rows];
                    if val.is_nan() {
                        if omit_nan {
                            continue;
                        }
                    } else {
                        saw_value = true;
                        if val == 0.0 {
                            all_true = false;
                            break;
                        }
                        continue;
                    }
                }
                if omit_nan && !saw_value {
                    all_true = true;
                }
                v[c] = if all_true { 1.0 } else { 0.0 };
            }
            (v, vec![1, cols])
        } else if dim == 1 {
            let mut v = vec![0.0f64; rows];
            for r in 0..rows {
                let mut all_true = true;
                let mut saw_value = false;
                for c in 0..cols {
                    let val = data[r + c * rows];
                    if val.is_nan() {
                        if omit_nan {
                            continue;
                        }
                    } else {
                        saw_value = true;
                        if val == 0.0 {
                            all_true = false;
                            break;
                        }
                        continue;
                    }
                }
                if omit_nan && !saw_value {
                    all_true = true;
                }
                v[r] = if all_true { 1.0 } else { 0.0 };
            }
            (v, vec![rows, 1])
        } else {
            return Err(anyhow!("reduce_all_dim: invalid dimension {}", dim));
        };
        Ok(self.allocate_tensor(out, shape))
    }

    fn reduce_median(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?
                .clone()
        };
        let median = if data.is_empty() || data.iter().any(|v| v.is_nan()) {
            f64::NAN
        } else {
            let mut scratch = data.clone();
            compute_median_inplace(&mut scratch)
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![median]);
        Ok(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_median_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("reduce_median_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let mut scratch = Vec::<f64>::with_capacity(rows.max(cols));
        let out = if dim <= 1 {
            let mut v = vec![f64::NAN; cols];
            for c in 0..cols {
                scratch.clear();
                let mut saw_nan = false;
                for r in 0..rows {
                    let val = abuf[r + c * rows];
                    if val.is_nan() {
                        saw_nan = true;
                        scratch.clear();
                        break;
                    }
                    scratch.push(val);
                }
                v[c] = if saw_nan || scratch.is_empty() {
                    f64::NAN
                } else {
                    compute_median_inplace(&mut scratch)
                };
            }
            v
        } else {
            let mut v = vec![f64::NAN; rows];
            for r in 0..rows {
                scratch.clear();
                let mut saw_nan = false;
                for c in 0..cols {
                    let val = abuf[r + c * rows];
                    if val.is_nan() {
                        saw_nan = true;
                        scratch.clear();
                        break;
                    }
                    scratch.push(val);
                }
                v[r] = if saw_nan || scratch.is_empty() {
                    f64::NAN
                } else {
                    compute_median_inplace(&mut scratch)
                };
            }
            v
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, out);
        let shape = if dim <= 1 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        Ok(GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_min(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let m = abuf.iter().cloned().fold(f64::INFINITY, f64::min);
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![m]);
        Ok(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_min_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
    ) -> Result<runmat_accelerate_api::ReduceDimResult> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("reduce_min_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let (vals, inds, vshape) = if dim <= 1 {
            let mut m: Vec<f64> = vec![f64::INFINITY; cols];
            let mut idx: Vec<f64> = vec![1.0; cols];
            for c in 0..cols {
                for r in 0..rows {
                    let v = abuf[r + c * rows];
                    if v < m[c] {
                        m[c] = v;
                        idx[c] = (r + 1) as f64;
                    }
                }
            }
            (m, idx, vec![1, cols])
        } else {
            let mut m: Vec<f64> = vec![f64::INFINITY; rows];
            let mut idx: Vec<f64> = vec![1.0; rows];
            for r in 0..rows {
                for c in 0..cols {
                    let v = abuf[r + c * rows];
                    if v < m[r] {
                        m[r] = v;
                        idx[r] = (c + 1) as f64;
                    }
                }
            }
            (m, idx, vec![rows, 1])
        };
        drop(guard);
        let idv = self.next_id.fetch_add(1, Ordering::Relaxed);
        let idi = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut g = registry().lock().unwrap();
        g.insert(idv, vals);
        g.insert(idi, inds);
        let shape_vals = vshape.clone();
        let shape_inds = vshape;
        Ok(runmat_accelerate_api::ReduceDimResult {
            values: GpuTensorHandle {
                shape: shape_vals,
                device_id: 0,
                buffer_id: idv,
            },
            indices: GpuTensorHandle {
                shape: shape_inds,
                device_id: 0,
                buffer_id: idi,
            },
        })
    }

    fn reduce_max(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let m = abuf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![m]);
        Ok(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn reduce_max_dim(
        &self,
        a: &GpuTensorHandle,
        dim: usize,
    ) -> Result<runmat_accelerate_api::ReduceDimResult> {
        if a.shape.len() != 2 {
            return Err(anyhow::anyhow!("reduce_max_dim: only 2D supported"));
        }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let (vals, inds, vshape) = if dim <= 1 {
            let mut m: Vec<f64> = vec![f64::NEG_INFINITY; cols];
            let mut idx: Vec<f64> = vec![1.0; cols];
            for c in 0..cols {
                for r in 0..rows {
                    let v = abuf[r + c * rows];
                    if v > m[c] {
                        m[c] = v;
                        idx[c] = (r + 1) as f64;
                    }
                }
            }
            (m, idx, vec![1, cols])
        } else {
            let mut m: Vec<f64> = vec![f64::NEG_INFINITY; rows];
            let mut idx: Vec<f64> = vec![1.0; rows];
            for r in 0..rows {
                for c in 0..cols {
                    let v = abuf[r + c * rows];
                    if v > m[r] {
                        m[r] = v;
                        idx[r] = (c + 1) as f64;
                    }
                }
            }
            (m, idx, vec![rows, 1])
        };
        drop(guard);
        let idv = self.next_id.fetch_add(1, Ordering::Relaxed);
        let idi = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut g = registry().lock().unwrap();
        g.insert(idv, vals);
        g.insert(idi, inds);
        let shape_vals = vshape.clone();
        let shape_inds = vshape;
        Ok(runmat_accelerate_api::ReduceDimResult {
            values: GpuTensorHandle {
                shape: shape_vals,
                device_id: 0,
                buffer_id: idv,
            },
            indices: GpuTensorHandle {
                shape: shape_inds,
                device_id: 0,
                buffer_id: idi,
            },
        })
    }

    fn cumsum_scan(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> Result<GpuTensorHandle> {
        Err(anyhow!("cumsum_scan not supported by provider"))
    }

    fn cummin_scan(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> Result<runmat_accelerate_api::ProviderCumminResult> {
        Err(anyhow!("cummin_scan not supported by provider"))
    }

    fn find(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        let shape = a.shape.clone();
        let row_extent = shape.first().copied().unwrap_or(1).max(1);
        let (indices, rows, cols, values) = {
            let guard = registry().lock().unwrap();
            let data = guard
                .get(&a.buffer_id)
                .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
            let total = data.len();
            let cap = match direction {
                FindDirection::First => limit.unwrap_or(total),
                FindDirection::Last => limit.unwrap_or(1),
            }
            .min(total);

            let mut indices = Vec::new();
            let mut rows_out = Vec::new();
            let mut cols_out = Vec::new();
            let mut values_out = Vec::new();

            if cap == 0 || total == 0 {
                (indices, rows_out, cols_out, values_out)
            } else {
                match direction {
                    FindDirection::First => {
                        for (idx, &value) in data.iter().enumerate() {
                            if value != 0.0 {
                                indices.push((idx + 1) as f64);
                                rows_out.push(((idx % row_extent) + 1) as f64);
                                cols_out.push(((idx / row_extent) + 1) as f64);
                                values_out.push(value);
                                if indices.len() >= cap {
                                    break;
                                }
                            }
                        }
                    }
                    FindDirection::Last => {
                        for (idx, &value) in data.iter().enumerate().rev() {
                            if value != 0.0 {
                                indices.push((idx + 1) as f64);
                                rows_out.push(((idx % row_extent) + 1) as f64);
                                cols_out.push(((idx / row_extent) + 1) as f64);
                                values_out.push(value);
                                if indices.len() >= cap {
                                    break;
                                }
                            }
                        }
                    }
                }
                (indices, rows_out, cols_out, values_out)
            }
        };

        let count = indices.len();
        let shape_out = vec![count, 1];
        let linear = self.allocate_tensor(indices, shape_out.clone());
        let rows_handle = self.allocate_tensor(rows, shape_out.clone());
        let cols_handle = self.allocate_tensor(cols, shape_out.clone());
        let values_handle = self.allocate_tensor(values, shape_out);

        Ok(ProviderFindResult {
            linear,
            rows: rows_handle,
            cols: cols_handle,
            values: Some(values_handle),
        })
    }

    fn lu(&self, a: &GpuTensorHandle) -> Result<ProviderLuResult> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("lu: unknown buffer {}", a.buffer_id))?
        };
        let LuHostFactors {
            combined,
            lower,
            upper,
            perm_matrix,
            pivot_vector,
            combined_shape,
            lower_shape,
            upper_shape,
            perm_shape,
            pivot_shape,
        } = lu_factor_host(&data, &a.shape)?;
        let combined = self.allocate_tensor(combined, combined_shape);
        let lower = self.allocate_tensor(lower, lower_shape);
        let upper = self.allocate_tensor(upper, upper_shape);
        let perm_matrix = self.allocate_tensor(perm_matrix, perm_shape);
        let perm_vector = self.allocate_tensor(pivot_vector, pivot_shape);
        Ok(ProviderLuResult {
            combined,
            lower,
            upper,
            perm_matrix,
            perm_vector,
        })
    }

    fn chol(&self, a: &GpuTensorHandle, lower: bool) -> Result<ProviderCholResult> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&a.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("chol: unknown buffer {}", a.buffer_id))?
        };
        let tensor = Tensor::new(data, a.shape.clone()).map_err(|e| anyhow!("chol: {e}"))?;
        let mut args = Vec::new();
        if lower {
            args.push(Value::from("lower"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::chol::evaluate(
            Value::Tensor(tensor),
            &args,
        )
        .map_err(|err| runtime_flow_to_anyhow("chol", err))?;
        let factor_tensor = tensor_from_value("chol", eval.factor())?;
        let factor = self.allocate_tensor(factor_tensor.data.clone(), factor_tensor.shape.clone());
        Ok(ProviderCholResult {
            factor,
            info: eval.flag_index() as u32,
        })
    }

    fn qr(&self, handle: &GpuTensorHandle, options: ProviderQrOptions) -> Result<ProviderQrResult> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&handle.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("qr: unknown buffer {}", handle.buffer_id))?
        };
        let tensor = Tensor::new(data, handle.shape.clone()).map_err(|e| anyhow!("qr: {e}"))?;
        let mut args = Vec::new();
        if options.economy {
            args.push(Value::Num(0.0));
        }
        if matches!(options.pivot, ProviderQrPivot::Vector) {
            args.push(Value::from("vector"));
        }
        let eval = runmat_runtime::builtins::math::linalg::factor::qr::evaluate(
            Value::Tensor(tensor),
            &args,
        )
        .map_err(|err| runtime_flow_to_anyhow("qr", err))?;

        let q_tensor = tensor_from_value("qr", eval.q())?;
        let r_tensor = tensor_from_value("qr", eval.r())?;
        let perm_matrix_tensor = tensor_from_value("qr", eval.permutation_matrix())?;
        let perm_vector_tensor = tensor_from_value("qr", eval.permutation_vector())?;

        let q_handle = self.allocate_tensor(q_tensor.data.clone(), q_tensor.shape.clone());
        let r_handle = self.allocate_tensor(r_tensor.data.clone(), r_tensor.shape.clone());
        let perm_matrix_handle = self.allocate_tensor(
            perm_matrix_tensor.data.clone(),
            perm_matrix_tensor.shape.clone(),
        );
        let perm_vector_handle = self.allocate_tensor(
            perm_vector_tensor.data.clone(),
            perm_vector_tensor.shape.clone(),
        );

        Ok(ProviderQrResult {
            q: q_handle,
            r: r_handle,
            perm_matrix: perm_matrix_handle,
            perm_vector: perm_vector_handle,
        })
    }

    fn matmul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        // Only support 2D shapes for reference provider
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(anyhow::anyhow!("matmul: only 2D supported"));
        }
        let (ar, ac) = (a.shape[0], a.shape[1]);
        let (br, bc) = (b.shape[0], b.shape[1]);
        if ac != br {
            return Err(anyhow::anyhow!("matmul: inner dims must agree"));
        }
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        let mut out = vec![0.0; ar * bc];
        // Column-major multiplication
        for j in 0..bc {
            for i in 0..ar {
                let mut sum = 0.0;
                for k in 0..ac {
                    sum += abuf[i + k * ar] * bbuf[k + j * br];
                }
                out[i + j * ar] = sum;
            }
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: vec![ar, bc],
            device_id: 0,
            buffer_id: id,
        })
    }

    fn matmul_epilogue(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        ep: &runmat_accelerate_api::MatmulEpilogue,
    ) -> Result<GpuTensorHandle> {
        // Compute plain matmul first
        let base = self.matmul(a, b)?;
        let (rows, cols) = (base.shape[0], base.shape[1]);
        let mut data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&base.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("matmul_epilogue: unknown buffer {}", base.buffer_id))?
        };

        // Load optional scales from registry
        let row_scale: Option<Vec<f64>> = if let Some(ref h) = ep.row_scale {
            let guard = registry().lock().unwrap();
            Some(guard.get(&h.buffer_id).cloned().ok_or_else(|| {
                anyhow!("matmul_epilogue: unknown row scale buffer {}", h.buffer_id)
            })?)
        } else {
            None
        };
        let col_scale: Option<Vec<f64>> = if let Some(ref h) = ep.col_scale {
            let guard = registry().lock().unwrap();
            Some(guard.get(&h.buffer_id).cloned().ok_or_else(|| {
                anyhow!("matmul_epilogue: unknown col scale buffer {}", h.buffer_id)
            })?)
        } else {
            None
        };
        let mut diag_output: Option<Vec<f64>> = if let Some(ref h) = ep.diag_output {
            let guard = registry().lock().unwrap();
            let vec = guard.get(&h.buffer_id).cloned().ok_or_else(|| {
                anyhow!(
                    "matmul_epilogue: unknown diag output buffer {}",
                    h.buffer_id
                )
            })?;
            Some(vec)
        } else {
            None
        };
        if let Some(ref diag) = diag_output {
            let expected = rows.min(cols);
            if diag.len() < expected {
                return Err(anyhow!(
                    "matmul_epilogue: diag_output length {} insufficient for diag size {}",
                    diag.len(),
                    expected
                ));
            }
        }

        // Apply epilogue: alpha/beta, then per-row/col scales (matches GPU epilogue ordering)
        for j in 0..cols {
            for i in 0..rows {
                let idx = i + j * rows;
                let mut v = data[idx] * ep.alpha + ep.beta;
                if let Some(ref rs) = row_scale {
                    let s = *rs.get(i).unwrap_or(&1.0);
                    v = match ep.row_op {
                        runmat_accelerate_api::ScaleOp::Multiply => v * s,
                        runmat_accelerate_api::ScaleOp::Divide => v / s,
                    };
                }
                if let Some(ref cs) = col_scale {
                    let s = *cs.get(j).unwrap_or(&1.0);
                    v = match ep.col_op {
                        runmat_accelerate_api::ScaleOp::Multiply => v * s,
                        runmat_accelerate_api::ScaleOp::Divide => v / s,
                    };
                }
                if let Some(min_v) = ep.clamp_min {
                    v = v.max(min_v);
                }
                if let Some(max_v) = ep.clamp_max {
                    v = v.min(max_v);
                }
                if let Some(pow_v) = ep.pow_exponent {
                    v = v.powf(pow_v);
                }
                if let Some(ref mut diag) = diag_output {
                    if i == j && i < diag.len() {
                        diag[i] = v;
                    }
                }
                data[idx] = v;
            }
        }

        // Replace buffer contents with epilogued data
        {
            let mut guard = registry().lock().unwrap();
            guard.insert(base.buffer_id, data);
            if let Some(vec) = diag_output {
                if let Some(ref h) = ep.diag_output {
                    guard.insert(h.buffer_id, vec);
                }
            }
        }
        Ok(base)
    }

    fn matmul_power_step(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        ep: &runmat_accelerate_api::PowerStepEpilogue,
    ) -> Result<GpuTensorHandle> {
        let base = self.matmul(lhs, rhs)?;
        let rows = base.shape[0];
        let cols = base.shape[1];
        let mut data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&base.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("matmul_power_step: unknown buffer {}", base.buffer_id))?
        };
        let mut norms = vec![0.0f64; cols];
        for (col, norm) in norms.iter_mut().enumerate().take(cols) {
            let mut acc = 0.0f64;
            for row in 0..rows {
                let idx = row + col * rows;
                let val = data[idx];
                acc += val * val;
            }
            acc += ep.epsilon;
            *norm = acc.sqrt();
        }
        for (col, norm) in norms.iter().enumerate().take(cols) {
            for row in 0..rows {
                let idx = row + col * rows;
                data[idx] /= norm;
            }
        }
        {
            let mut guard = registry().lock().unwrap();
            guard.insert(base.buffer_id, data);
        }
        Ok(base)
    }

    fn image_normalize(
        &self,
        input: &GpuTensorHandle,
        desc: &runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            input.shape.len() == 3,
            "image_normalize: expected 3-D tensor, got {:?}",
            input.shape
        );
        ensure!(
            input.shape[0] == desc.batch
                && input.shape[1] == desc.height
                && input.shape[2] == desc.width,
            "image_normalize: descriptor dims {:?} do not match tensor shape {:?}",
            (desc.batch, desc.height, desc.width),
            input.shape
        );

        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&input.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("image_normalize: unknown buffer {}", input.buffer_id))?
        };

        let batch = desc.batch;
        let height = desc.height;
        let width = desc.width;
        let plane = height * width;
        if plane == 0 {
            return Ok(self.allocate_tensor(vec![], input.shape.clone()));
        }

        let stride_h = batch;
        let stride_w = batch * height;

        let gain = desc.gain.unwrap_or(1.0);
        let bias = desc.bias.unwrap_or(0.0);
        let gamma = desc.gamma;

        let mut output = data.clone();

        for b in 0..batch {
            let mut sum = 0.0;
            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    sum += data[idx];
                }
            }
            let mean = sum / plane as f64;

            let mut sq_sum = 0.0;
            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    let diff = data[idx] - mean;
                    sq_sum += diff * diff;
                }
            }
            let variance = sq_sum / plane as f64;
            let sigma = (variance + desc.epsilon).sqrt();
            let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };

            for w in 0..width {
                let base_w = w * stride_w;
                for h in 0..height {
                    let idx = b + h * stride_h + base_w;
                    let mut value = (data[idx] - mean) * inv_sigma;
                    if desc.gain.is_some() {
                        value *= gain;
                    }
                    if desc.bias.is_some() {
                        value += bias;
                    }
                    value = value.max(0.0);
                    if let Some(gamma) = gamma {
                        value = value.powf(gamma);
                    }
                    output[idx] = value;
                }
            }
        }

        Ok(self.allocate_tensor(output, input.shape.clone()))
    }

    fn pagefun(&self, _request: &PagefunRequest) -> Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "pagefun: in-process provider does not implement GPU page operations"
        ))
    }

    fn linsolve(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        options: &ProviderLinsolveOptions,
    ) -> Result<ProviderLinsolveResult> {
        let (lhs_data, rhs_data) = {
            let guard = registry().lock().unwrap();
            let lhs_buf = guard
                .get(&lhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("linsolve: unknown buffer {}", lhs.buffer_id))?;
            let rhs_buf = guard
                .get(&rhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("linsolve: unknown buffer {}", rhs.buffer_id))?;
            (lhs_buf, rhs_buf)
        };

        let lhs_tensor =
            Tensor::new(lhs_data, lhs.shape.clone()).map_err(|e| anyhow!("linsolve: {e}"))?;
        let rhs_tensor =
            Tensor::new(rhs_data, rhs.shape.clone()).map_err(|e| anyhow!("linsolve: {e}"))?;

        let (solution, rcond) = linsolve_host_real_for_provider(&lhs_tensor, &rhs_tensor, options)
            .map_err(|e| anyhow!("{e}"))?;

        let Tensor { data, shape, .. } = solution;
        let handle = self.allocate_tensor(data, shape);
        Ok(ProviderLinsolveResult {
            solution: handle,
            reciprocal_condition: rcond,
        })
    }

    fn inv(
        &self,
        matrix: &GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("inv: unknown buffer {}", matrix.buffer_id))?
        };
        let tensor = Tensor::new(data, matrix.shape.clone()).map_err(|e| anyhow!("inv: {e}"))?;
        let result = inv_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
        let Tensor { data, shape, .. } = result;
        Ok(self.allocate_tensor(data, shape))
    }

    fn pinv(
        &self,
        matrix: &GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("pinv: unknown buffer {}", matrix.buffer_id))?
        };
        let tensor = Tensor::new(data, matrix.shape.clone()).map_err(|e| anyhow!("pinv: {e}"))?;
        let result =
            pinv_host_real_for_provider(&tensor, options.tolerance).map_err(|e| anyhow!("{e}"))?;
        let Tensor { data, shape, .. } = result;
        Ok(self.allocate_tensor(data, shape))
    }

    fn cond(&self, matrix: &GpuTensorHandle, norm: ProviderCondNorm) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("cond: unknown buffer {}", matrix.buffer_id))?
        };
        let tensor = Tensor::new(data, matrix.shape.clone()).map_err(|e| anyhow!("cond: {e}"))?;
        let cond_value = cond_host_real_for_provider(&tensor, norm).map_err(|e| anyhow!("{e}"))?;
        Ok(self.allocate_tensor(vec![cond_value], vec![1, 1]))
    }

    fn norm(&self, tensor: &GpuTensorHandle, order: ProviderNormOrder) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&tensor.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("norm: unknown buffer {}", tensor.buffer_id))?
        };
        let host_tensor =
            Tensor::new(data, tensor.shape.clone()).map_err(|e| anyhow!("norm: {e}"))?;
        let value = norm_host_real_for_provider(&host_tensor, order).map_err(|e| anyhow!("{e}"))?;
        Ok(self.allocate_tensor(vec![value], vec![1, 1]))
    }

    fn rank(&self, matrix: &GpuTensorHandle, tolerance: Option<f64>) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("rank: unknown buffer {}", matrix.buffer_id))?
        };

        let tensor = Tensor::new(data, matrix.shape.clone()).map_err(|e| anyhow!("rank: {e}"))?;
        let rank =
            rank_host_real_for_provider(&tensor, tolerance).map_err(|e| anyhow!("{e}"))? as f64;

        Ok(self.allocate_tensor(vec![rank], vec![1, 1]))
    }

    fn rcond(&self, matrix: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let data = {
            let guard = registry().lock().unwrap();
            guard
                .get(&matrix.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("rcond: unknown buffer {}", matrix.buffer_id))?
        };
        let tensor = Tensor::new(data, matrix.shape.clone()).map_err(|e| anyhow!("rcond: {e}"))?;
        let estimate = rcond_host_real_for_provider(&tensor).map_err(|e| anyhow!("{e}"))?;
        Ok(self.allocate_tensor(vec![estimate], vec![1, 1]))
    }

    fn mldivide(&self, lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (lhs_data, rhs_data) = {
            let guard = registry().lock().unwrap();
            let lhs_buf = guard
                .get(&lhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("mldivide: unknown buffer {}", lhs.buffer_id))?;
            let rhs_buf = guard
                .get(&rhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("mldivide: unknown buffer {}", rhs.buffer_id))?;
            (lhs_buf, rhs_buf)
        };

        let lhs_tensor =
            Tensor::new(lhs_data, lhs.shape.clone()).map_err(|e| anyhow!("mldivide: {e}"))?;
        let rhs_tensor =
            Tensor::new(rhs_data, rhs.shape.clone()).map_err(|e| anyhow!("mldivide: {e}"))?;

        let result = mldivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;

        let Tensor { data, shape, .. } = result;
        Ok(self.allocate_tensor(data, shape))
    }

    fn mrdivide(&self, lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let (lhs_data, rhs_data) = {
            let guard = registry().lock().unwrap();
            let lhs_buf = guard
                .get(&lhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("mrdivide: unknown buffer {}", lhs.buffer_id))?;
            let rhs_buf = guard
                .get(&rhs.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow!("mrdivide: unknown buffer {}", rhs.buffer_id))?;
            (lhs_buf, rhs_buf)
        };

        let lhs_tensor =
            Tensor::new(lhs_data, lhs.shape.clone()).map_err(|e| anyhow!("mrdivide: {e}"))?;
        let rhs_tensor =
            Tensor::new(rhs_data, rhs.shape.clone()).map_err(|e| anyhow!("mrdivide: {e}"))?;

        let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
            .map_err(|e| anyhow!("{e}"))?;

        let Tensor { data, shape, .. } = result;
        Ok(self.allocate_tensor(data, shape))
    }

    fn eig(&self, _a: &GpuTensorHandle, _compute_left: bool) -> Result<ProviderEigResult> {
        Err(anyhow!("eig: not supported by in-process provider"))
    }

    fn sub2ind(
        &self,
        dims: &[usize],
        strides: &[usize],
        inputs: &[&GpuTensorHandle],
        scalar_mask: &[bool],
        len: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        if inputs.len() != dims.len() || inputs.len() != scalar_mask.len() {
            return Err(anyhow::anyhow!(
                "sub2ind: expected {} subscripts for {} dimensions",
                dims.len(),
                dims.len()
            ));
        }
        let expected_len: usize = output_shape.iter().copied().product();
        if expected_len != len {
            return Err(anyhow::anyhow!(
                "sub2ind: output shape does not match subscript sizes"
            ));
        }
        if len == 0 {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            registry().lock().unwrap().insert(id, Vec::new());
            return Ok(GpuTensorHandle {
                shape: output_shape.to_vec(),
                device_id: 0,
                buffer_id: id,
            });
        }

        let mut host_values: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());
        {
            let guard = registry().lock().unwrap();
            for handle in inputs {
                let data = guard
                    .get(&handle.buffer_id)
                    .ok_or_else(|| anyhow::anyhow!("sub2ind: unknown buffer {}", handle.buffer_id))?
                    .clone();
                host_values.push(data);
            }
        }

        let mut output = Vec::with_capacity(len);
        for idx in 0..len {
            let mut offset: usize = 0;
            for (dim_index, ((&dim_size, &stride), data)) in dims
                .iter()
                .zip(strides.iter())
                .zip(host_values.iter())
                .enumerate()
            {
                let raw = if scalar_mask[dim_index] {
                    *data.first().unwrap_or(&0.0)
                } else {
                    data[idx]
                };
                let coerced = coerce_sub2ind_value(raw, dim_index + 1, dim_size)?;
                let term = coerced
                    .checked_sub(1)
                    .and_then(|base| base.checked_mul(stride))
                    .ok_or_else(|| {
                        anyhow::anyhow!("sub2ind: computed index exceeds platform limits")
                    })?;
                offset = offset.checked_add(term).ok_or_else(|| {
                    anyhow::anyhow!("sub2ind: computed index exceeds platform limits")
                })?;
            }
            output.push((offset + 1) as f64);
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, output);
        Ok(GpuTensorHandle {
            shape: output_shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn fused_elementwise(
        &self,
        _shader: &str,
        _inputs: &[GpuTensorHandle],
        _output_shape: &[usize],
        _len: usize,
    ) -> Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "fused_elementwise not supported by in-process provider"
        ))
    }
}

static INSTANCE: OnceCell<InProcessProvider> = OnceCell::new();

/// Register the in-process provider as the global acceleration provider.
/// Safe to call multiple times; only the first call installs the provider.
pub fn register_inprocess_provider() {
    let provider: &'static InProcessProvider = INSTANCE.get_or_init(InProcessProvider::new);
    // Safety: we intentionally install a reference with 'static lifetime. Always reassert.
    unsafe { runmat_accelerate_api::register_provider(provider) };
}

/// Reset the in-process provider RNG to its default seed (test-only helper).
pub fn reset_inprocess_rng() {
    if let Ok(mut guard) = rng_state().lock() {
        *guard = 0x9e3779b97f4a7c15;
    }
}
