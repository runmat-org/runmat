use crate::builtins::common::tensor;
use runmat_accelerate_api::HostTensorOwned;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use std::collections::HashSet;

/// Parse the optional FFT length argument, returning `None` for `[]`.
pub fn parse_length(value: &Value, builtin: &str) -> Result<Option<usize>, String> {
    match value {
        Value::Tensor(t) if t.data.is_empty() => Ok(None),
        Value::ComplexTensor(t) if t.data.is_empty() => Ok(None),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err(format!("{builtin}: length must be a scalar"));
            }
            parse_length_scalar(t.data[0], builtin).map(Some)
        }
        Value::ComplexTensor(t) => {
            if t.data.len() != 1 {
                return Err(format!("{builtin}: length must be a scalar"));
            }
            let (re, im) = t.data[0];
            if im.abs() > f64::EPSILON {
                return Err(format!("{builtin}: length must be real-valued"));
            }
            parse_length_scalar(re, builtin).map(Some)
        }
        Value::Num(n) => parse_length_scalar(*n, builtin).map(Some),
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(format!("{builtin}: length must be non-negative"));
            }
            Ok(Some(raw as usize))
        }
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(format!("{builtin}: length must be real-valued"));
            }
            parse_length_scalar(*re, builtin).map(Some)
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            Err(format!("{builtin}: length must be numeric"))
        }
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::GpuTensor(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Symbolic(_) => Err(format!("{builtin}: length must be numeric")),
    }
}

fn parse_length_scalar(value: f64, builtin: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!("{builtin}: length must be finite"));
    }
    if value < 0.0 {
        return Err(format!("{builtin}: length must be non-negative"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!("{builtin}: length must be an integer"));
    }
    Ok(rounded as usize)
}

/// Convert any numeric value into a `ComplexTensor`.
pub fn value_to_complex_tensor(value: Value, builtin: &str) -> Result<ComplexTensor, String> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Tensor(tensor) => tensor_to_complex_tensor(tensor, builtin),
        Value::Num(n) => {
            ComplexTensor::new(vec![(n, 0.0)], vec![1, 1]).map_err(|e| format!("{builtin}: {e}"))
        }
        Value::Int(i) => {
            let val = i.to_f64();
            ComplexTensor::new(vec![(val, 0.0)], vec![1, 1]).map_err(|e| format!("{builtin}: {e}"))
        }
        Value::Bool(b) => {
            let val = if b { 1.0 } else { 0.0 };
            ComplexTensor::new(vec![(val, 0.0)], vec![1, 1]).map_err(|e| format!("{builtin}: {e}"))
        }
        Value::Complex(re, im) => {
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("{builtin}: {e}"))
        }
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .and_then(|t| tensor_to_complex_tensor(t, builtin))
            .map_err(|e| format!("{builtin}: {e}")),
        other => Err(format!(
            "{builtin}: unsupported input type {:?}; expected numeric or complex data",
            other
        )),
    }
}

/// Convert a real-valued tensor into a `ComplexTensor`.
pub fn tensor_to_complex_tensor(tensor: Tensor, builtin: &str) -> Result<ComplexTensor, String> {
    let data = tensor
        .data
        .into_iter()
        .map(|re| (re, 0.0))
        .collect::<Vec<_>>();
    ComplexTensor::new(data, tensor.shape).map_err(|e| format!("{builtin}: {e}"))
}

/// Convert a downloaded host tensor into a complex tensor, interpreting a trailing
/// dimension of size 2 as `[real, imag]` pairs.
pub fn host_to_complex_tensor(
    host: HostTensorOwned,
    builtin: &str,
) -> Result<ComplexTensor, String> {
    let HostTensorOwned { data, shape } = host;
    if shape.last() == Some(&2) {
        if data.len() % 2 != 0 {
            return Err(format!(
                "{builtin}: provider tensor has mismatched real/imag data"
            ));
        }
        let mut complex_shape = shape;
        complex_shape.pop();
        if complex_shape.is_empty() {
            complex_shape.push(1);
        }
        let mut complex_data = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            complex_data.push((chunk[0], chunk[1]));
        }
        ComplexTensor::new(complex_data, complex_shape).map_err(|e| format!("{builtin}: {e}"))
    } else {
        let tensor = Tensor::new(data, shape).map_err(|e| format!("{builtin}: {e}"))?;
        tensor_to_complex_tensor(tensor, builtin)
    }
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

/// Kind of circular shift that should be applied to match MATLAB semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    /// `fftshift` semantics (floor(N/2)).
    Fft,
    /// `ifftshift` semantics (ceil(N/2)).
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

    let mut ext_shape = shape.to_vec();
    if dims.iter().copied().max().unwrap_or(0) >= ext_shape.len() {
        let needed = dims.iter().copied().max().unwrap_or(0) + 1;
        while ext_shape.len() < needed {
            ext_shape.push(1);
        }
    }

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
    data: &[T],
    shape: &[usize],
    shifts: &[usize],
) -> Result<Vec<T>, String> {
    if shape.len() != shifts.len() {
        return Err("shift: internal shape mismatch".to_string());
    }
    let total: usize = shape.iter().product();
    if total != data.len() {
        return Err("shift: shape does not match data length".to_string());
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

/// Compute the zero-based dimension indices to shift.
pub fn compute_shift_dims(
    shape: &[usize],
    arg: Option<&Value>,
    builtin: &str,
) -> Result<Vec<usize>, String> {
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
                return Err(format!("{builtin}: dimension indices must be >= 1"));
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

fn dims_from_value(value: &Value, builtin: &str) -> Result<Vec<usize>, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 1 {
                return Err(format!("{builtin}: dimension indices must be >= 1"));
            }
            Ok(vec![raw as usize])
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!("{builtin}: dimensions must be finite integers"));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(format!("{builtin}: dimensions must be integers"));
            }
            if rounded < 1.0 {
                return Err(format!("{builtin}: dimension indices must be >= 1"));
            }
            Ok(vec![rounded as usize])
        }
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) && !tensor.data.is_empty() {
                return Err(format!(
                    "{builtin}: dimension vectors must be row or column vectors"
                ));
            }
            let mut dims = Vec::with_capacity(tensor.data.len());
            for &val in &tensor.data {
                if !val.is_finite() {
                    return Err(format!("{builtin}: dimensions must be finite integers"));
                }
                let rounded = val.round();
                if (rounded - val).abs() > f64::EPSILON {
                    return Err(format!("{builtin}: dimensions must be integers"));
                }
                if rounded < 1.0 {
                    return Err(format!("{builtin}: dimension indices must be >= 1"));
                }
                dims.push(rounded as usize);
            }
            Ok(dims)
        }
        Value::LogicalArray(array) => {
            if !is_vector_shape(&array.shape) && !array.data.is_empty() {
                return Err(format!(
                    "{builtin}: dimension masks must be row or column vectors"
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
        Value::GpuTensor(_) => Err(format!(
            "{builtin}: dimension specification must reside on the host"
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Symbolic(_) => Err(format!("{builtin}: dimension indices must be numeric")),
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}
