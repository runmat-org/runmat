use crate::builtins::common::tensor;
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use num_complex::Complex;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorOwned};
use runmat_builtins::{ComplexTensor, IntValue, NumericStorage, Tensor, Value};
use rustfft::FftPlanner;
use std::borrow::Cow;
use std::collections::HashSet;
use std::mem::size_of;
use std::sync::Arc;

fn builtin_error(builtin: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

fn checked_product(dims: &[usize], builtin: &str, context: &str) -> BuiltinResult<usize> {
    dims.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            builtin_error(
                builtin,
                format!("{builtin}: {context} dimension product overflow"),
            )
        })
    })
}

fn checked_mul(a: usize, b: usize, builtin: &str, context: &str) -> BuiltinResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| builtin_error(builtin, format!("{builtin}: {context} overflow")))
}

fn zeroed_complex_vec(
    len: usize,
    builtin: &str,
    context: &str,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let max_len = isize::MAX as usize / size_of::<Complex<f64>>();
    if len > max_len {
        return Err(builtin_error(
            builtin,
            format!("{builtin}: {context} allocation is too large"),
        ));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|err| {
        builtin_error(
            builtin,
            format!("{builtin}: {context} allocation failed: {err}"),
        )
    })?;
    values.resize(len, Complex::new(0.0, 0.0));
    Ok(values)
}

/// Parse the optional FFT length argument, returning `None` for `[]`.
pub fn parse_length(value: &Value, builtin: &str) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Tensor(t) if tensor_len(t) == 0 => Ok(None),
        Value::ComplexTensor(t) if t.data.is_empty() => Ok(None),
        Value::Tensor(t) => {
            if tensor_len(t) != 1 {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: length must be a scalar"),
                ));
            }
            if let Some(storage) = t.integer_storage() {
                return storage
                    .value_at(0)
                    .expect("one-element integer storage")
                    .try_to_usize()
                    .map(Some)
                    .ok_or_else(|| {
                        builtin_error(builtin, format!("{builtin}: length must be non-negative"))
                    });
            }
            parse_length_scalar(tensor::tensor_value_f64(t, 0), builtin).map(Some)
        }
        Value::ComplexTensor(t) => {
            if t.data.len() != 1 {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: length must be a scalar"),
                ));
            }
            let (re, im) = t.data[0];
            if im.abs() > f64::EPSILON {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: length must be real-valued"),
                ));
            }
            parse_length_scalar(re, builtin).map(Some)
        }
        Value::Num(n) => parse_length_scalar(*n, builtin).map(Some),
        Value::Int(i) => i.try_to_usize().map(Some).ok_or_else(|| {
            builtin_error(builtin, format!("{builtin}: length must be non-negative"))
        }),
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: length must be real-valued"),
                ));
            }
            parse_length_scalar(*re, builtin).map(Some)
        }
        Value::Bool(_) | Value::LogicalArray(_) => Err(builtin_error(
            builtin,
            format!("{builtin}: length must be numeric"),
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::GpuTensor(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(builtin_error(
            builtin,
            format!("{builtin}: length must be numeric"),
        )),
    }
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn parse_length_scalar(value: f64, builtin: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(builtin_error(
            builtin,
            format!("{builtin}: length must be finite"),
        ));
    }
    if value < 0.0 {
        return Err(builtin_error(
            builtin,
            format!("{builtin}: length must be non-negative"),
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(builtin_error(
            builtin,
            format!("{builtin}: length must be an integer"),
        ));
    }
    Ok(rounded as usize)
}

/// Convert any numeric value into a `ComplexTensor`.
pub fn value_to_complex_tensor(value: Value, builtin: &str) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Tensor(tensor) => tensor_to_complex_tensor(tensor, builtin),
        Value::Num(n) => ComplexTensor::new(vec![(n, 0.0)], vec![1, 1])
            .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}"))),
        Value::Int(i) => {
            let val = i.to_f64();
            ComplexTensor::new(vec![(val, 0.0)], vec![1, 1])
                .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))
        }
        Value::Bool(b) => {
            let val = if b { 1.0 } else { 0.0 };
            ComplexTensor::new(vec![(val, 0.0)], vec![1, 1])
                .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))
        }
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}"))),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?;
            tensor_to_complex_tensor(tensor, builtin)
        }
        other => Err(builtin_error(
            builtin,
            format!(
                "{builtin}: unsupported input type {:?}; expected numeric or complex data",
                other
            ),
        )),
    }
}

/// Convert a real-valued tensor into a `ComplexTensor`.
pub fn tensor_to_complex_tensor(tensor: Tensor, builtin: &str) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    // FFT arithmetic currently uses double complex storage. Materialize the
    // authoritative real input explicitly at this transform-domain boundary.
    let data = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?
        .materialize_f64()
        .into_iter()
        .map(|re| (re, 0.0))
        .collect::<Vec<_>>();
    ComplexTensor::new(data, shape).map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))
}

pub fn complex_tensor_to_real_value(tensor: ComplexTensor, builtin: &str) -> BuiltinResult<Value> {
    let data = tensor.data.iter().map(|(re, _)| *re).collect::<Vec<_>>();
    let real = Tensor::new(data, tensor.shape.clone())
        .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?;
    Ok(Value::Tensor(real))
}

/// Convert a downloaded host tensor into a complex tensor, interpreting a trailing
/// dimension of size 2 as `[real, imag]` pairs.
pub fn host_to_complex_tensor(
    host: HostTensorOwned,
    builtin: &str,
) -> BuiltinResult<ComplexTensor> {
    let HostTensorOwned {
        data,
        shape,
        storage,
    } = host;
    if storage == GpuTensorStorage::ComplexInterleaved {
        if data.len() % 2 != 0 {
            return Err(builtin_error(
                builtin,
                format!("{builtin}: provider tensor has mismatched real/imag data"),
            ));
        }
        let mut complex_shape = shape;
        if complex_shape.is_empty() {
            complex_shape.push(1);
        }
        let mut complex_data = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            complex_data.push((chunk[0], chunk[1]));
        }
        ComplexTensor::new(complex_data, complex_shape)
            .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))
    } else {
        let tensor = Tensor::new(data, shape)
            .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?;
        tensor_to_complex_tensor(tensor, builtin)
    }
}

pub async fn gather_gpu_complex_tensor(
    handle: &GpuTensorHandle,
    builtin: &str,
) -> BuiltinResult<ComplexTensor> {
    let provider = runmat_accelerate_api::provider_for_handle(handle)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| {
            builtin_error(
                builtin,
                format!("{builtin}: no acceleration provider registered"),
            )
        })?;
    let host = download_handle_async(provider, handle)
        .await
        .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?;
    host_to_complex_tensor(host, builtin)
}

pub async fn download_provider_complex_tensor(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    builtin: &str,
    free_after_download: bool,
) -> BuiltinResult<ComplexTensor> {
    let host = download_handle_async(provider, handle)
        .await
        .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?;
    if free_after_download {
        provider.free(handle).ok();
    }
    runmat_accelerate_api::clear_residency(handle);
    host_to_complex_tensor(host, builtin)
}

/// Return the first non-singleton dimension (1-based), defaulting to 1.
pub fn default_dimension(shape: &[usize]) -> usize {
    for (idx, &dim) in shape.iter().enumerate() {
        if dim != 1 {
            return idx + 1;
        }
    }
    1
}

/// Remove trailing singleton dimensions while keeping at least `minimum_rank` axes.
pub fn trim_trailing_ones(shape: &mut Vec<usize>, minimum_rank: usize) {
    while shape.len() > minimum_rank && shape.last() == Some(&1) {
        shape.pop();
    }
    if shape.is_empty() {
        shape.push(1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformDirection {
    Forward,
    Inverse,
}

pub fn transform_complex_tensor(
    mut tensor: ComplexTensor,
    length: Option<usize>,
    dimension: Option<usize>,
    direction: TransformDirection,
    builtin: &str,
) -> BuiltinResult<ComplexTensor> {
    let origin_rank = tensor.shape.len();
    if crate::builtins::common::shape::is_scalar_shape(&tensor.shape) {
        tensor.shape = crate::builtins::common::shape::normalize_scalar_shape(&tensor.shape);
        tensor.rows = tensor.shape.first().copied().unwrap_or(1);
        tensor.cols = tensor.shape.get(1).copied().unwrap_or(1);
    }

    let mut shape = tensor.shape.clone();
    let dim_index = match dimension {
        Some(0) => {
            return Err(builtin_error(
                builtin,
                format!("{builtin}: dimension must be >= 1"),
            ))
        }
        Some(dim) => dim - 1,
        None => default_dimension(&shape) - 1,
    };

    while shape.len() <= dim_index {
        shape.push(1);
    }

    let current_len = shape[dim_index];
    let target_len = length.unwrap_or(current_len);
    if target_len == 0 {
        let mut out_shape = shape;
        out_shape[dim_index] = 0;
        trim_trailing_ones(&mut out_shape, origin_rank);
        return ComplexTensor::new(Vec::<(f64, f64)>::new(), out_shape)
            .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")));
    }

    let inner_stride = checked_product(&shape[..dim_index], builtin, "inner stride")?;
    let outer_stride = checked_product(&shape[dim_index + 1..], builtin, "outer stride")?;
    let num_slices = checked_mul(inner_stride, outer_stride, builtin, "slice count")?;

    let input = tensor
        .data
        .into_iter()
        .map(|(re, im)| Complex::new(re, im))
        .collect::<Vec<_>>();

    if num_slices == 0 {
        let mut out_shape = shape;
        out_shape[dim_index] = target_len;
        trim_trailing_ones(&mut out_shape, origin_rank.max(dim_index + 1));
        return ComplexTensor::new(Vec::<(f64, f64)>::new(), out_shape)
            .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")));
    }

    let output_len = checked_mul(target_len, num_slices, builtin, "output length")?;
    let mut output = zeroed_complex_vec(output_len, builtin, "output")?;

    let mut planner = FftPlanner::<f64>::new();
    let plan: Option<Arc<dyn rustfft::Fft<f64>>> = if target_len > 1 {
        Some(match direction {
            TransformDirection::Forward => planner.plan_fft_forward(target_len),
            TransformDirection::Inverse => planner.plan_fft_inverse(target_len),
        })
    } else {
        None
    };

    let copy_len = current_len.min(target_len);
    let mut buffer = zeroed_complex_vec(target_len, builtin, "workspace")?;
    let scale = match direction {
        TransformDirection::Forward => 1.0,
        TransformDirection::Inverse => 1.0 / (target_len as f64),
    };

    for outer in 0..outer_stride {
        let base_in = checked_mul(
            outer,
            checked_mul(current_len, inner_stride, builtin, "input slice span")?,
            builtin,
            "input slice offset",
        )?;
        let base_out = checked_mul(
            outer,
            checked_mul(target_len, inner_stride, builtin, "output slice span")?,
            builtin,
            "output slice offset",
        )?;
        for inner in 0..inner_stride {
            buffer.fill(Complex::new(0.0, 0.0));
            for (k, slot) in buffer.iter_mut().enumerate().take(copy_len) {
                let src_idx = base_in + inner + k * inner_stride;
                if src_idx < input.len() {
                    *slot = input[src_idx];
                }
            }
            if let Some(p) = &plan {
                p.process(&mut buffer);
            }
            for (k, value) in buffer.iter().enumerate().take(target_len) {
                let dst_idx = base_out + inner + k * inner_stride;
                if dst_idx < output.len() {
                    output[dst_idx] = *value * scale;
                }
            }
        }
    }

    let mut out_shape = shape;
    out_shape[dim_index] = target_len;
    trim_trailing_ones(&mut out_shape, origin_rank.max(dim_index + 1));

    let data = output.into_iter().map(|c| (c.re, c.im)).collect::<Vec<_>>();
    ComplexTensor::new(data, out_shape)
        .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))
}

pub fn transform_nd_complex_tensor(
    mut tensor: ComplexTensor,
    sizes: Option<&[usize]>,
    direction: TransformDirection,
    builtin: &str,
) -> BuiltinResult<ComplexTensor> {
    let axis_count = sizes
        .map(|v| v.len())
        .unwrap_or_else(|| tensor.shape.len().max(1));
    for axis in 0..axis_count {
        let len = sizes.and_then(|v| v.get(axis).copied());
        tensor = transform_complex_tensor(tensor, len, Some(axis + 1), direction, builtin)?;
    }
    Ok(tensor)
}

pub fn transform_axes_complex_tensor(
    mut tensor: ComplexTensor,
    lengths: &[Option<usize>],
    direction: TransformDirection,
    builtin: &str,
) -> BuiltinResult<ComplexTensor> {
    for (axis, &len) in lengths.iter().enumerate() {
        tensor = transform_complex_tensor(tensor, len, Some(axis + 1), direction, builtin)?;
    }
    Ok(tensor)
}

pub fn parse_2d_lengths_from_data(
    data: &[f64],
    builtin: &str,
) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match data.len() {
        0 => Ok((None, None)),
        1 => {
            let scalar = Value::Num(data[0]);
            let len = parse_length(&scalar, builtin)?;
            Ok((len, len))
        }
        2 => {
            let first = Value::Num(data[0]);
            let second = Value::Num(data[1]);
            let len_rows = parse_length(&first, builtin)?;
            let len_cols = parse_length(&second, builtin)?;
            Ok((len_rows, len_cols))
        }
        _ => Err(builtin_error(
            builtin,
            format!("{builtin}: size vector must contain at most two elements"),
        )),
    }
}

pub fn parse_2d_lengths_from_tensor(
    tensor: &Tensor,
    builtin: &str,
) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    if let Some(storage) = tensor.integer_storage() {
        let exact = storage.exact_values();
        return match exact.len() {
            0 => Ok((None, None)),
            1 => {
                let len = parse_length(&Value::Int(exact[0].clone()), builtin)?;
                Ok((len, len))
            }
            2 => {
                let len_rows = parse_length(&Value::Int(exact[0].clone()), builtin)?;
                let len_cols = parse_length(&Value::Int(exact[1].clone()), builtin)?;
                Ok((len_rows, len_cols))
            }
            _ => Err(builtin_error(
                builtin,
                format!("{builtin}: size vector must contain at most two elements"),
            )),
        };
    }
    parse_2d_lengths_from_data(tensor::tensor_values_f64_cow(tensor).as_ref(), builtin)
}

pub fn parse_nd_sizes_value(value: &Value, builtin: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return parse_nd_sizes_integers(&storage.exact_values(), builtin);
            }
            parse_nd_sizes_data(tensor::tensor_values_f64_cow(t).as_ref(), builtin)
        }
        Value::LogicalArray(logical) => {
            let t = tensor::logical_to_tensor(logical)
                .map_err(|e| builtin_error(builtin, format!("{builtin}: {e}")))?;
            parse_nd_sizes_data(tensor::tensor_values_f64_cow(&t).as_ref(), builtin)
        }
        Value::Num(n) => parse_nd_sizes_data(&[*n], builtin),
        Value::Int(i) => parse_nd_sizes_integers(std::slice::from_ref(i), builtin),
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: SIZE must be real-valued"),
                ));
            }
            parse_nd_sizes_data(&[*re], builtin)
        }
        Value::ComplexTensor(_) => Err(builtin_error(
            builtin,
            format!("{builtin}: SIZE must be real-valued"),
        )),
        _ => Err(builtin_error(
            builtin,
            format!("{builtin}: SIZE must be numeric"),
        )),
    }
}

fn parse_nd_sizes_data(data: &[f64], builtin: &str) -> BuiltinResult<Vec<usize>> {
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        if !v.is_finite() {
            return Err(builtin_error(
                builtin,
                format!("{builtin}: SIZE values must be finite"),
            ));
        }
        if v < 0.0 {
            return Err(builtin_error(
                builtin,
                format!("{builtin}: SIZE values must be non-negative"),
            ));
        }
        let rounded = v.round();
        if (rounded - v).abs() > f64::EPSILON {
            return Err(builtin_error(
                builtin,
                format!("{builtin}: SIZE values must be integers"),
            ));
        }
        if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
            return Err(builtin_error(
                builtin,
                format!("{builtin}: SIZE values exceed the maximum supported size"),
            ));
        }
        out.push(rounded as usize);
    }
    Ok(out)
}

fn parse_nd_sizes_integers(data: &[IntValue], builtin: &str) -> BuiltinResult<Vec<usize>> {
    data.iter()
        .map(|value| {
            value.try_to_usize().ok_or_else(|| {
                builtin_error(
                    builtin,
                    format!("{builtin}: SIZE values must be non-negative platform integers"),
                )
            })
        })
        .collect()
}

pub fn parse_symflag(value: &Value, builtin: &str) -> BuiltinResult<Option<bool>> {
    let text: Option<Cow<'_, str>> = match value {
        Value::String(s) => Some(Cow::Borrowed(s.as_str())),
        Value::CharArray(ca) if ca.rows == 1 => {
            let collected: String = ca.data.iter().collect();
            Some(Cow::Owned(collected))
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Some(Cow::Borrowed(sa.data[0].as_str())),
        _ => None,
    };

    let Some(text) = text else {
        return Ok(None);
    };

    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("symmetric") {
        Ok(Some(true))
    } else if trimmed.eq_ignore_ascii_case("nonsymmetric") {
        Ok(Some(false))
    } else {
        Err(builtin_error(
            builtin,
            format!("{builtin}: unrecognized option '{trimmed}'"),
        ))
    }
}

/// Kind of circular shift that should be applied to match MATLAB semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    /// `fftshift` semantics (./2)).
    Fft,
    /// `ifftshift` semantics (./2)).
    Ifft,
}

/// Shift plan describing the reshaped tensor and offsets to apply.
#[derive(Debug, Clone)]
pub struct ShiftPlan {
    pub ext_shape: Vec<usize>,
    pub positive: Vec<usize>,
    pub provider: Vec<isize>,
}

impl ShiftPlan {
    #[inline]
    pub fn is_noop(&self) -> bool {
        self.positive.iter().all(|&shift| shift == 0)
    }
}

/// Build a shift plan for the supplied shape, dimensions, and shift kind.
pub fn build_shift_plan(shape: &[usize], dims: &[usize], kind: ShiftKind) -> ShiftPlan {
    if dims.is_empty() {
        return ShiftPlan {
            ext_shape: shape.to_vec(),
            positive: vec![0; shape.len()],
            provider: vec![0; shape.len()],
        };
    }

    let ext_shape = shape.to_vec();
    let mut positive = vec![0usize; ext_shape.len()];
    for &axis in dims {
        if axis >= ext_shape.len() {
            continue;
        }
        let size = ext_shape[axis];
        if size > 1 {
            positive[axis] = shift_amount(size, kind);
        }
    }

    let provider = positive.iter().map(|&v| v as isize).collect::<Vec<_>>();

    ShiftPlan {
        ext_shape,
        positive,
        provider,
    }
}

fn shift_amount(size: usize, kind: ShiftKind) -> usize {
    if size <= 1 {
        return 0;
    }
    match kind {
        ShiftKind::Fft => size / 2,
        ShiftKind::Ifft => size.div_ceil(2),
    }
}

/// Apply a circular shift to the provided data.
pub fn apply_shift<T: Clone>(
    builtin: &str,
    data: &[T],
    shape: &[usize],
    shifts: &[usize],
) -> BuiltinResult<Vec<T>> {
    if shape.len() != shifts.len() {
        return Err(builtin_error(
            builtin,
            format!("{builtin}: internal shape mismatch"),
        ));
    }
    let total: usize = shape.iter().product();
    if total != data.len() {
        return Err(builtin_error(
            builtin,
            format!("{builtin}: shape does not match data length"),
        ));
    }
    if total == 0 || shifts.iter().all(|&s| s == 0) {
        return Ok(data.to_vec());
    }

    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }

    let mut result = vec![data[0].clone(); total];
    let mut coords = vec![0usize; shape.len()];

    for (dest_idx, dest_slot) in result.iter_mut().enumerate() {
        let mut remainder = dest_idx;
        for (coord, &size) in coords.iter_mut().zip(shape.iter()) {
            *coord = if size == 0 { 0 } else { remainder % size };
            if size > 0 {
                remainder /= size;
            }
        }
        let mut src_idx = 0usize;
        for (axis, ((&coord, &size), &stride)) in coords
            .iter()
            .zip(shape.iter())
            .zip(strides.iter())
            .enumerate()
        {
            if size == 0 {
                continue;
            }
            if size <= 1 || shifts[axis] == 0 {
                src_idx += coord * stride;
            } else {
                let shift = shifts[axis] % size;
                let src_coord = (coord + size - shift) % size;
                src_idx += src_coord * stride;
            }
        }
        *dest_slot = data[src_idx].clone();
    }

    Ok(result)
}

pub fn apply_shift_numeric_storage(
    builtin: &str,
    storage: NumericStorage,
    shape: &[usize],
    shifts: &[usize],
) -> BuiltinResult<NumericStorage> {
    macro_rules! shift {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(apply_shift(builtin, &$values, shape, shifts)?)
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => shift!(values, F64),
        NumericStorage::F32(values) => shift!(values, F32),
        NumericStorage::I8(values) => shift!(values, I8),
        NumericStorage::I16(values) => shift!(values, I16),
        NumericStorage::I32(values) => shift!(values, I32),
        NumericStorage::I64(values) => shift!(values, I64),
        NumericStorage::U8(values) => shift!(values, U8),
        NumericStorage::U16(values) => shift!(values, U16),
        NumericStorage::U32(values) => shift!(values, U32),
        NumericStorage::U64(values) => shift!(values, U64),
    })
}

/// Compute the zero-based dimension indices to shift.
pub fn compute_shift_dims(
    shape: &[usize],
    arg: Option<&Value>,
    builtin: &str,
) -> BuiltinResult<Vec<usize>> {
    let rank = if shape.is_empty() { 1 } else { shape.len() };
    if let Some(value) = arg {
        let dims1 = dims_from_value(value, builtin)?;
        if dims1.is_empty() {
            return Ok(Vec::new());
        }
        let mut dims = Vec::with_capacity(dims1.len());
        let mut seen = HashSet::new();
        for dim in dims1 {
            if dim == 0 {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: dimension indices must be >= 1"),
                ));
            }
            let zero_based = dim - 1;
            if seen.insert(zero_based) {
                dims.push(zero_based);
            }
        }
        Ok(dims)
    } else {
        Ok((0..rank).collect())
    }
}

fn dims_from_value(value: &Value, builtin: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(i) => i
            .try_to_usize()
            .filter(|dimension| *dimension >= 1)
            .map(|dimension| vec![dimension])
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    format!("{builtin}: dimension indices must be >= 1"),
                )
            }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: dimensions must be finite integers"),
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: dimensions must be integers"),
                ));
            }
            if rounded < 1.0 {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: dimension indices must be >= 1"),
                ));
            }
            Ok(vec![rounded as usize])
        }
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) && !tensor.is_empty() {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: dimension vectors must be row or column vectors"),
                ));
            }
            if let Some(parsed) = tensor::integer_tensor_dimension_vector(tensor, builtin, false) {
                return parsed.map_err(|_| {
                    builtin_error(
                        builtin,
                        format!("{builtin}: dimension indices must be >= 1"),
                    )
                });
            }
            let values = tensor::tensor_values_f64_cow(tensor);
            let mut dims = Vec::with_capacity(values.len());
            for &val in values.iter() {
                if !val.is_finite() {
                    return Err(builtin_error(
                        builtin,
                        format!("{builtin}: dimensions must be finite integers"),
                    ));
                }
                let rounded = val.round();
                if (rounded - val).abs() > f64::EPSILON {
                    return Err(builtin_error(
                        builtin,
                        format!("{builtin}: dimensions must be integers"),
                    ));
                }
                if rounded < 1.0 {
                    return Err(builtin_error(
                        builtin,
                        format!("{builtin}: dimension indices must be >= 1"),
                    ));
                }
                dims.push(rounded as usize);
            }
            Ok(dims)
        }
        Value::LogicalArray(array) => {
            if !is_vector_shape(&array.shape) && !array.data.is_empty() {
                return Err(builtin_error(
                    builtin,
                    format!("{builtin}: dimension masks must be row or column vectors"),
                ));
            }
            let mut dims = Vec::new();
            for (idx, &flag) in array.data.iter().enumerate() {
                if flag != 0 {
                    dims.push(idx + 1);
                }
            }
            Ok(dims)
        }
        Value::GpuTensor(_) => Err(builtin_error(
            builtin,
            format!("{builtin}: dimension specification must reside on the host"),
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::SparseTensor(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(builtin_error(
            builtin,
            format!("{builtin}: dimension indices must be numeric"),
        )),
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn fft_length_preserves_typed_integer_scalar_tensors() {
        #[cfg(target_pointer_width = "64")]
        {
            let exact_value = 9_007_199_254_740_993_u64;
            let mut exact =
                Tensor::new_integer(IntegerStorage::U64(vec![exact_value]), vec![1, 1]).unwrap();
            exact.data.clear();
            assert_eq!(
                parse_length(&Value::Tensor(exact), "fft").unwrap(),
                Some(exact_value as usize)
            );
        }

        let err = parse_length(
            &Value::Tensor(Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap()),
            "fft",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("non-negative"), "unexpected error: {err}");
    }

    #[test]
    fn fft_single_inputs_materialize_at_the_current_double_complex_boundary() {
        let input = Tensor::from_f32(vec![0.1, -2.0], vec![1, 2]).unwrap();
        let complex = tensor_to_complex_tensor(input, "fft").expect("complex input");
        assert_eq!(complex.data, vec![(f64::from(0.1_f32), 0.0), (-2.0, 0.0)]);

        let length = Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap();
        assert_eq!(
            parse_length(&Value::Tensor(length), "fft").unwrap(),
            Some(4)
        );
    }

    #[test]
    fn fft_2d_lengths_preserve_typed_integer_size_vectors() {
        #[cfg(target_pointer_width = "64")]
        {
            let exact = 9_007_199_254_740_993_u64;
            let mut sizes =
                Tensor::new_integer(IntegerStorage::U64(vec![1, exact]), vec![1, 2]).unwrap();
            sizes.data.fill(f64::NAN);

            assert_eq!(
                parse_2d_lengths_from_tensor(&sizes, "fft2").unwrap(),
                (Some(1), Some(exact as usize))
            );
        }

        let negative = Tensor::new_integer(IntegerStorage::I16(vec![4, -1]), vec![1, 2]).unwrap();
        let err = parse_2d_lengths_from_tensor(&negative, "fft2")
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-negative"), "unexpected error: {err}");
    }

    #[test]
    fn nd_sizes_preserve_typed_integer_values_and_bounds() {
        let sizes = Tensor::new_integer(
            IntegerStorage::U64(vec![1, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(
            parse_nd_sizes_value(&Value::Tensor(sizes), "fftn").unwrap(),
            vec![1, 9_007_199_254_740_993usize]
        );

        let err = parse_nd_sizes_value(&Value::Int(IntValue::I64(-1)), "fftn")
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-negative"), "unexpected error: {err}");

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = parse_nd_sizes_value(&Value::Num(boundary), "fftn")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("maximum supported size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn shift_dims_preserve_typed_integer_vectors_without_rank_extension() {
        #[cfg(target_pointer_width = "64")]
        {
            let exact_dim = 9_007_199_254_740_993_u64;
            let dims =
                Tensor::new_integer(IntegerStorage::U64(vec![1, exact_dim]), vec![1, 2]).unwrap();
            assert_eq!(
                compute_shift_dims(&[2, 2], Some(&Value::Tensor(dims)), "fftshift").unwrap(),
                vec![0, exact_dim as usize - 1]
            );

            let plan = build_shift_plan(&[2, 2], &[exact_dim as usize - 1], ShiftKind::Fft);
            assert_eq!(plan.ext_shape, vec![2, 2]);
            assert_eq!(plan.positive, vec![0, 0]);
            assert_eq!(
                apply_shift(
                    "fftshift",
                    &[1.0, 2.0, 3.0, 4.0],
                    &plan.ext_shape,
                    &plan.positive
                )
                .unwrap(),
                vec![1.0, 2.0, 3.0, 4.0]
            );
        }
    }
}
