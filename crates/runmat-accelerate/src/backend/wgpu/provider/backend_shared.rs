use super::*;

fn runtime_flow_to_anyhow(_context: &str, err: RuntimeError) -> anyhow::Error {
    anyhow::Error::new(err)
}

fn validate_compute_binding_counts(
    operation: &str,
    storage_bindings: usize,
    total_bindings: usize,
    limits: &wgpu::Limits,
) -> Result<()> {
    let storage_limit = limits.max_storage_buffers_per_shader_stage as usize;
    ensure!(
        storage_bindings <= storage_limit,
        "{}: requires {} storage buffers, but this WebGPU adapter supports {} per shader stage",
        operation,
        storage_bindings,
        storage_limit
    );

    let binding_limit = limits.max_bindings_per_bind_group as usize;
    ensure!(
        total_bindings <= binding_limit,
        "{}: requires {} bind group entries, but this WebGPU adapter supports {} per bind group",
        operation,
        total_bindings,
        binding_limit
    );

    Ok(())
}

fn checked_binding_count(operation: &str, left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{}: binding count overflow", operation))
}

fn gpu_per_buffer_limit_error(
    operation: &str,
    requested_bytes: u64,
    max_bytes: u64,
) -> anyhow::Error {
    let requested_mib = requested_bytes as f64 / (1024.0 * 1024.0);
    let max_mib = max_bytes as f64 / (1024.0 * 1024.0);
    anyhow!(
        "{operation}: requested {requested_bytes} bytes ({requested_mib:.2} MiB) exceeds this device per-buffer limit of {max_bytes} bytes ({max_mib:.2} MiB). This is a per-buffer backend limit (not total VRAM). Split the data into smaller arrays/chunks and process iteratively."
    )
}

fn gpu_dispatch_length_limit_error(operation: &str, len: usize) -> anyhow::Error {
    anyhow!(
        "{operation}: tensor length {len} exceeds the current GPU kernel indexing limit of {} elements. Split the operation into smaller chunks and process iteratively.",
        u32::MAX
    )
}

#[cfg(test)]
mod compute_binding_count_tests {
    use super::{checked_binding_count, validate_compute_binding_counts, WgpuProvider};

    #[test]
    fn rejects_storage_bindings_over_adapter_stage_limit() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            ..Default::default()
        };

        let err = validate_compute_binding_counts("fused_elementwise_multi", 11, 12, &limits)
            .expect_err(
                "storage binding overflow should be rejected before creating a WGPU layout",
            );

        assert!(err.to_string().contains("requires 11 storage buffers"));
    }

    #[test]
    fn accepts_bindings_at_adapter_limits() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            max_bindings_per_bind_group: 11,
            ..Default::default()
        };

        validate_compute_binding_counts("fused_elementwise", 10, 11, &limits)
            .expect("limits are inclusive");
    }

    #[test]
    fn rejects_bindings_over_bind_group_limit() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 10,
            max_bindings_per_bind_group: 11,
            ..Default::default()
        };

        let err = validate_compute_binding_counts("fused_elementwise", 10, 12, &limits)
            .expect_err("bind group entry overflow should be rejected");

        assert!(err.to_string().contains("requires 12 bind group entries"));
    }

    #[test]
    fn rejects_binding_count_overflow() {
        let err = checked_binding_count("fused_elementwise", usize::MAX, 1)
            .expect_err("binding count overflow should be rejected");

        assert!(err.to_string().contains("binding count overflow"));
    }

    #[test]
    fn poolable_bytes_uses_default_and_caps_to_adapter_limit() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(None, 0),
            256u64 << 20
        );
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(None, 128u64 << 20),
            128u64 << 20
        );
    }

    #[test]
    fn poolable_bytes_honors_env_override_and_adapter_cap() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(
                Some("1073741824"),
                512u64 << 20
            ),
            512u64 << 20
        );
    }

    #[test]
    fn poolable_bytes_accepts_zero_to_disable_pooling() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(Some("0"), 2u64 << 30),
            0
        );
    }

    #[test]
    fn poolable_bytes_invalid_override_falls_back_to_default() {
        assert_eq!(
            WgpuProvider::parse_buffer_residency_max_poolable_bytes(Some("bad"), 512u64 << 20),
            256u64 << 20
        );
    }
}


fn parse_two_pass_mode(raw: &str) -> Option<ReductionTwoPassMode> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "auto" => Some(ReductionTwoPassMode::Auto),
        "force_on" | "on" | "true" | "1" => Some(ReductionTwoPassMode::ForceOn),
        "force_off" | "off" | "false" | "0" => Some(ReductionTwoPassMode::ForceOff),
        _ => None,
    }
}

fn build_matrix_operand_view(
    handle: &GpuTensorHandle,
    entry: &BufferEntry,
) -> Result<MatrixOperandView> {
    if entry.shape.len() < 2 {
        return Err(anyhow!(
            "matrix operand requires at least 2D tensor (buffer {} shape {:?})",
            handle.buffer_id,
            entry.shape
        ));
    }
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
        if rows != info.base_cols || cols != info.base_rows {
            return Err(anyhow!(
                "transpose metadata mismatch for buffer {}",
                handle.buffer_id
            ));
        }
        let lda = u32::try_from(info.base_rows)
            .map_err(|_| anyhow!("leading dimension exceeds GPU limits"))?;
        Ok(MatrixOperandView {
            rows,
            cols,
            lda,
            transpose: true,
        })
    } else {
        let lda =
            u32::try_from(rows).map_err(|_| anyhow!("leading dimension exceeds GPU limits"))?;
        Ok(MatrixOperandView {
            rows,
            cols,
            lda,
            transpose: false,
        })
    }
}

fn canonical_vendor_name(info: &wgpu::AdapterInfo) -> String {
    match info.vendor {
        0x10DE => "NVIDIA".to_string(),
        0x1002 | 0x1022 => "AMD".to_string(),
        0x8086 => "Intel".to_string(),
        0x106B => "Apple".to_string(),
        0x13B5 => "ARM".to_string(),
        0x5143 => "Qualcomm".to_string(),
        0x1414 => "Microsoft".to_string(),
        0x1AE0 => "Google".to_string(),
        0x1C5C => "Huawei".to_string(),
        0 => info
            .name
            .split_whitespace()
            .next()
            .unwrap_or("unknown")
            .to_string(),
        other => {
            let prefix = info.name.split_whitespace().next().unwrap_or("vendor");
            format!("{prefix} (0x{other:04x})")
        }
    }
}

const POLYDER_EPS: f64 = 1.0e-12;

#[derive(Clone, Copy)]
enum PolynomialOrientation {
    Scalar,
    Row,
    Column,
}

fn polynomial_orientation(shape: &[usize]) -> Result<PolynomialOrientation> {
    let mut non_unit = 0usize;
    let mut orientation = PolynomialOrientation::Scalar;
    for (idx, &dim) in shape.iter().enumerate() {
        if dim > 1 {
            non_unit += 1;
            orientation = if idx == 0 {
                PolynomialOrientation::Column
            } else {
                PolynomialOrientation::Row
            };
        }
    }
    if non_unit > 1 {
        Err(anyhow!(
            "polyder: coefficient tensors must be vectors on the GPU"
        ))
    } else {
        Ok(orientation)
    }
}

fn conv_orientation_for(orientation: PolynomialOrientation) -> ProviderConvOrientation {
    match orientation {
        PolynomialOrientation::Column => ProviderConvOrientation::Column,
        PolynomialOrientation::Scalar | PolynomialOrientation::Row => ProviderConvOrientation::Row,
    }
}

fn shape_for_orientation(orientation: PolynomialOrientation, len: usize) -> Vec<usize> {
    if len <= 1 {
        return vec![1, 1];
    }
    match orientation {
        PolynomialOrientation::Scalar | PolynomialOrientation::Row => vec![1, len],
        PolynomialOrientation::Column => vec![len, 1],
    }
}

fn trim_leading_zeros_real(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    if let Some(idx) = coeffs.iter().position(|c| c.abs() > POLYDER_EPS) {
        coeffs[idx..].to_vec()
    } else {
        vec![0.0]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolyderParams {
    input_len: u32,
    output_len: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolyintParamsF64 {
    input_len: u32,
    output_len: u32,
    constant: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolyintParamsF32 {
    input_len: u32,
    output_len: u32,
    constant: f32,
    _pad0: f32,
}

fn normalize_eye_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => normalize_scalar_shape(shape),
        1 => {
            let n = shape[0];
            normalize_scalar_shape(&[n, n])
        }
        _ => normalize_scalar_shape(shape),
    }
}

fn normalize_concat_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    if shape.is_empty() {
        return normalize_scalar_shape(&shape);
    }
    let min_len = ((dim_zero + 1).max(2)).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    normalize_scalar_shape(&shape)
}

fn normalize_gradient_shape(shape: &[usize], len: usize) -> Vec<usize> {
    matlab_gradient_shape(shape, len)
}

fn conv1d_output_shape(len: usize, orientation: ProviderConvOrientation) -> Vec<usize> {
    match (orientation, len) {
        (ProviderConvOrientation::Row, 0) => vec![1, 0],
        (ProviderConvOrientation::Row, _) => vec![1, len],
        (ProviderConvOrientation::Column, 0) => vec![0, 1],
        (ProviderConvOrientation::Column, _) => vec![len, 1],
    }
}

fn conv1d_window(
    signal_len: usize,
    kernel_len: usize,
    mode: ProviderConvMode,
) -> Result<(usize, usize, usize)> {
    if signal_len == 0 || kernel_len == 0 {
        return Ok((0, 0, 0));
    }
    let full_len = signal_len
        .checked_add(kernel_len)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| anyhow!("conv1d: result length overflow"))?;
    let (output_len, start_offset) = match mode {
        ProviderConvMode::Full => (full_len, 0usize),
        ProviderConvMode::Same => {
            let start = if kernel_len == 0 {
                0
            } else {
                (kernel_len - 1) / 2
            };
            let len = signal_len.min(full_len.saturating_sub(start));
            (len, start)
        }
        ProviderConvMode::Valid => {
            if signal_len < kernel_len {
                (0usize, 0usize)
            } else {
                (signal_len - kernel_len + 1, kernel_len - 1)
            }
        }
    };
    if output_len == 0 {
        return Ok((0, start_offset, full_len));
    }
    ensure!(
        start_offset
            .checked_add(output_len)
            .map(|v| v <= full_len)
            .unwrap_or(false),
        "conv1d: window exceeds full convolution length"
    );
    Ok((output_len, start_offset, full_len))
}

fn product_checked(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn canonical_matrix_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => {
            let mut out = shape.to_vec();
            if out.len() == 1 {
                out.push(1);
            }
            out
        }
    }
}

fn pad_dims(mut dims: Vec<usize>, rank: usize) -> Vec<usize> {
    if dims.len() < rank {
        dims.resize(rank, 1);
    } else if dims.len() > rank {
        dims.truncate(rank);
    }
    dims
}

fn compute_page_strides(dims: &[usize]) -> Vec<usize> {
    let mut stride = 1usize;
    let mut out = Vec::with_capacity(dims.len());
    for &dim in dims {
        out.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    out
}

fn decode_multi_index(mut index: usize, dims: &[usize], out: &mut [usize]) {
    for (dim, &extent) in dims.iter().enumerate() {
        if extent == 0 {
            out[dim] = 0;
        } else {
            out[dim] = index % extent;
            index /= extent;
        }
    }
}

fn broadcast_linear_index(dims: &[usize], strides: &[usize], multi_index: &[usize]) -> usize {
    let mut linear = 0usize;
    for ((&extent, &stride), &coord) in dims.iter().zip(strides.iter()).zip(multi_index.iter()) {
        if extent == 0 {
            return 0;
        }
        let actual = if extent == 1 { 0 } else { coord };
        linear += actual * stride;
    }
    linear
}

fn gaussian_normalizer(rows: usize, cols: usize, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let denom = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for col in 0..cols {
        let dx = col as f64 - col_center;
        for row in 0..rows {
            let dy = row as f64 - row_center;
            sum += (-((dx * dx + dy * dy) / denom)).exp();
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        0.0
    } else {
        1.0 / sum
    }
}

fn shapes_compatible(expected: &[usize], actual: &[usize]) -> bool {
    let max_len = expected.len().max(actual.len());
    for idx in 0..max_len {
        let e = expected.get(idx).copied().unwrap_or(1);
        let a = actual.get(idx).copied().unwrap_or(1);
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

pub(crate) fn host_tensor_from_value(label: &str, value: Value) -> Result<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| anyhow!("{label}: {e}")),
        Value::Int(i) => {
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| anyhow!("{label}: {e}"))
        }
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| anyhow!("{label}: {e}")),
        Value::ComplexTensor(_) => Err(anyhow!(
            "{label}: complex outputs are not supported by the wgpu provider"
        )),
        other => Err(anyhow!("{label}: unexpected value {other:?}")),
    }
}

fn median_from_slice(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        let mut tmp = values.to_vec();
        compute_median_inplace(&mut tmp)
    }
}

fn diag_offset_abs(offset: isize) -> usize {
    if offset >= 0 {
        offset as usize
    } else {
        let magnitude = -(offset as i128);
        magnitude as usize
    }
}
fn diag_matrix_size_checked(len: usize, offset: isize) -> Result<(usize, usize)> {
    let shift = diag_offset_abs(offset);
    let size = len
        .checked_add(shift)
        .ok_or_else(|| anyhow!("diag: result dimension exceeds GPU limits"))?;
    let total = size
        .checked_mul(size)
        .ok_or_else(|| anyhow!("diag: result size exceeds GPU limits"))?;
    Ok((size, total))
}

fn diag_length(rows: usize, cols: usize, offset: isize) -> usize {
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
        let shift = diag_offset_abs(offset);
        if shift >= rows {
            0
        } else {
            (rows - shift).min(cols)
        }
    }
}

fn diag_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

fn diag_is_vector_like(rows: usize, cols: usize, dims: usize) -> bool {
    rows == 1 || cols == 1 || dims <= 1
}
fn diag_ensure_shape(shape: &[usize]) -> Result<()> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        Err(anyhow!("diag: input must be 2-D"))
    } else {
        Ok(())
    }
}

fn apply_tril_mask_host(data: &mut [f64], shape: &[usize], offset: isize) -> Result<()> {
    if data.is_empty() {
        return Ok(());
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
        return Ok(());
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
        return Ok(());
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
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            for row in 0..rows {
                if (row as isize) - (col as isize) < -offset {
                    data[col_base + row] = 0.0;
                }
            }
        }
    }
    Ok(())
}

fn apply_triu_mask_host(data: &mut [f64], shape: &[usize], offset: isize) -> Result<()> {
    if data.is_empty() {
        return Ok(());
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
        return Ok(());
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
        return Ok(());
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
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            let col_isize = col as isize;
            for row in 0..rows {
                let diff = col_isize - (row as isize);
                if diff < offset {
                    data[col_base + row] = 0.0;
                }
            }
        }
    }
    Ok(())
}

fn stride_before_for(shape: &[usize], dim: usize) -> usize {
    if dim == 0 {
        return 1;
    }
    let upper = dim.min(shape.len());
    shape[..upper]
        .iter()
        .copied()
        .fold(1usize, |acc, extent| acc.saturating_mul(extent.max(1)))
}

fn stride_after_for(shape: &[usize], dim: usize) -> usize {
    if dim + 1 >= shape.len() {
        return 1;
    }
    shape[(dim + 1)..]
        .iter()
        .copied()
        .fold(1usize, |acc, extent| acc.saturating_mul(extent.max(1)))
}
fn dimension_length_zero_based(shape: &[usize], dim: usize) -> usize {
    shape.get(dim).copied().unwrap_or(1)
}
fn compare_values_for_sort(
    a: f64,
    b: f64,
    order: SortOrder,
    comparison: SortComparison,
) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match order {
            SortOrder::Ascend => Ordering::Greater,
            SortOrder::Descend => Ordering::Less,
        },
        (false, true) => match order {
            SortOrder::Ascend => Ordering::Less,
            SortOrder::Descend => Ordering::Greater,
        },
        (false, false) => compare_finite_for_sort(a, b, order, comparison),
    }
}

fn compare_finite_for_sort(
    a: f64,
    b: f64,
    order: SortOrder,
    comparison: SortComparison,
) -> Ordering {
    let primary = if matches!(comparison, SortComparison::Abs) {
        let abs_cmp = a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal);
        if abs_cmp == Ordering::Equal {
            Ordering::Equal
        } else {
            match order {
                SortOrder::Ascend => abs_cmp,
                SortOrder::Descend => abs_cmp.reverse(),
            }
        }
    } else {
        Ordering::Equal
    };
    if primary != Ordering::Equal {
        return primary;
    }
    match order {
        SortOrder::Ascend => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        SortOrder::Descend => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

fn sort_host_tensor(
    data: &[f64],
    shape: &[usize],
    dim: usize,
    order: SortOrder,
    comparison: SortComparison,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let expected_len = if shape.is_empty() {
        1usize
    } else {
        product_checked(shape)
            .ok_or_else(|| anyhow!("sort_dim: tensor size exceeds supported limits"))?
    };
    ensure!(
        expected_len == data.len(),
        "sort_dim: tensor data length {} does not match shape {:?}",
        data.len(),
        shape
    );

    if data.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let dim_len = dimension_length_zero_based(shape, dim);
    let mut sorted = data.to_vec();
    let mut indices = if dim_len == 0 {
        Vec::new()
    } else {
        vec![1.0; sorted.len()]
    };

    if dim_len <= 1 {
        return Ok((sorted, indices));
    }

    let stride_before = stride_before_for(shape, dim);
    let stride_after = stride_after_for(shape, dim);
    let mut buffer: Vec<(usize, f64)> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let idx = before + k * stride_before + after * stride_before * dim_len;
                buffer.push((k, data[idx]));
            }
            buffer.sort_by(|a, b| compare_values_for_sort(a.1, b.1, order, comparison));
            for (pos, (original_index, value)) in buffer.iter().enumerate() {
                let target = before + pos * stride_before + after * stride_before * dim_len;
                sorted[target] = *value;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    Ok((sorted, indices))
}
const RNG_DEFAULT_SEED: u64 = 0x9e3779b97f4a7c15;
const MAX_SAFE_INTEGER: u64 = 1 << 53;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;

fn advance_rng_state(state: u64, mut delta: u64) -> u64 {
    let mut acc_mult = 1u64;
    let mut acc_plus = 0u64;
    let mut cur_mult = RNG_MULTIPLIER;
    let mut cur_plus = RNG_INCREMENT;

    while delta > 0 {
        if (delta & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_plus.wrapping_mul(cur_mult.wrapping_add(1));
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }

    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}
fn seed_from_state(state: u64) -> u32 {
    let high = (state >> 32) as u32;
    let low = state as u32;
    let mut seed = low ^ high.rotate_left(13);
    if seed == 0 {
        seed = 0x9E37_79B9;
    }
    seed | 1
}

fn philox_keys_from_state(state: u64) -> (u32, u32) {
    let lo = state as u32;
    let hi = (state >> 32) as u32;
    let mut key0 = lo ^ hi.rotate_left(7);
    if key0 == 0 {
        key0 = 0x9E37_79B9;
    }
    let mut key1 = hi ^ lo.rotate_right(3);
    if key1 == 0 {
        key1 = 0xBB67_AE85;
    }
    (key0, key1)
}

fn rng_state() -> &'static Mutex<u64> {
    static RNG: OnceCell<Mutex<u64>> = OnceCell::new();
    RNG.get_or_init(|| Mutex::new(RNG_DEFAULT_SEED))
}
static NEXT_SUBMISSION_ID: AtomicU32 = AtomicU32::new(1);

