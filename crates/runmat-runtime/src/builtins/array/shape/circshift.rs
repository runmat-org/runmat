//! MATLAB-compatible `circshift` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `circshift` function, matching MathWorks MATLAB
//! behaviour for rotating tensors, logical masks, complex arrays, string
//! arrays, and character matrices. When an acceleration provider exposes a
//! native `circshift` hook the runtime keeps data on the GPU; otherwise it
//! gathers the tensor once, performs the rotation on the host, and re-uploads
//! the result so downstream operations continue to benefit from gpu residency.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::arg_tokens::{tokens_from_values, ArgToken};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    CharArray, ComplexTensor, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;
use std::collections::HashSet;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::circshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "circshift",
    op_kind: GpuOpKind::Custom("circshift"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may implement a dedicated circshift hook; otherwise the runtime gathers, rotates, and re-uploads once.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::circshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "circshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Circshift reorders data; fusion planners treat it as a residency boundary between kernels.",
};

fn preserve_array_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape } => Type::Tensor { shape: shape.clone() },
        Type::Logical { shape } => Type::Logical { shape: shape.clone() },
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn circshift_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("circshift")
        .build()
}

#[runtime_builtin(
    name = "circshift",
    category = "array/shape",
    summary = "Rotate arrays circularly along one or more dimensions.",
    keywords = "circshift,circular shift,rotate array,gpu,cyclic shift",
    accel = "custom",
    type_resolver(preserve_array_type),
    builtin_path = "crate::builtins::array::shape::circshift"
)]
async fn circshift_builtin(
    value: Value,
    shift: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(circshift_error("circshift: too many input arguments"));
    }
    let spec = parse_circshift_spec(&shift, &rest)?;
    let dims = &spec.dims;
    let shifts = &spec.shifts;

    match value {
        Value::Tensor(tensor) => circshift_tensor(tensor, dims, shifts)
            .map(tensor::tensor_into_value)
            .map_err(Into::into),
        Value::LogicalArray(array) => circshift_logical_array(array, dims, shifts)
            .map(Value::LogicalArray)
            .map_err(Into::into),
        Value::ComplexTensor(ct) => circshift_complex_tensor(ct, dims, shifts)
            .map(Value::ComplexTensor)
            .map_err(Into::into),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| circshift_error(format!("circshift: {e}")))?;
            circshift_complex_tensor(tensor, dims, shifts)
                .map(complex_tensor_into_value)
                .map_err(Into::into)
        }
        Value::StringArray(strings) => circshift_string_array(strings, dims, shifts)
            .map(Value::StringArray)
            .map_err(Into::into),
        Value::CharArray(chars) => circshift_char_array(chars, dims, shifts).map_err(Into::into),
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("circshift", value)
                .map_err(|e| circshift_error(e))?;
            circshift_tensor(tensor, dims, shifts)
                .map(tensor::tensor_into_value)
                .map_err(Into::into)
        }
        Value::GpuTensor(handle) => circshift_gpu(handle, dims, shifts)
            .await
            .map_err(Into::into),
        Value::Cell(_) => Err(circshift_error(
            "circshift: cell arrays are not yet supported",
        )),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(circshift_error("circshift: unsupported input type")),
    }
}

#[derive(Debug, Clone)]
struct CircshiftSpec {
    dims: Vec<usize>,
    shifts: Vec<isize>,
}

fn parse_circshift_spec(shift: &Value, rest: &[Value]) -> crate::BuiltinResult<CircshiftSpec> {
    let shifts = value_to_shift_vector(shift)?;
    let dims: Vec<usize> = if rest.is_empty() {
        (0..shifts.len()).collect()
    } else {
        let tokens = tokens_from_values(rest);
        if let Some(token) = tokens.first() {
            if let Some(dims_vec) = dims_from_token(token) {
                if dims_vec.len() != shifts.len() {
                    return Err(circshift_error(
                        "circshift: shift and dimension vectors must have the same length",
                    ));
                }
                dims_vec.into_iter().map(|dim| dim - 1).collect()
            } else {
                let dims_vec = value_to_dims_vector(&rest[0])?;
                if dims_vec.len() != shifts.len() {
                    return Err(circshift_error(
                        "circshift: shift and dimension vectors must have the same length",
                    ));
                }
                dims_vec.into_iter().map(|dim| dim - 1).collect()
            }
        } else {
            let dims_vec = value_to_dims_vector(&rest[0])?;
            if dims_vec.len() != shifts.len() {
                return Err(circshift_error(
                    "circshift: shift and dimension vectors must have the same length",
                ));
            }
            dims_vec.into_iter().map(|dim| dim - 1).collect()
        }
    };

    if dims.len() != shifts.len() {
        return Err(circshift_error(
            "circshift: shift vector must match the number of dimensions",
        ));
    }

    let mut seen = HashSet::new();
    for &dim in &dims {
        if !seen.insert(dim) {
            return Err(circshift_error(
                "circshift: dimension indices must be unique",
            ));
        }
    }

    Ok(CircshiftSpec { dims, shifts })
}

fn dims_from_token(token: &ArgToken) -> Option<Vec<usize>> {
    match token {
        ArgToken::Number(value) => coerce_dim_value(*value).map(|dim| vec![dim]),
        ArgToken::Vector(values) => {
            if values.is_empty() {
                return None;
            }
            let mut dims = Vec::with_capacity(values.len());
            for value in values {
                let dim = match value {
                    ArgToken::Number(num) => coerce_dim_value(*num)?,
                    _ => return None,
                };
                dims.push(dim);
            }
            Some(dims)
        }
        _ => None,
    }
}

fn coerce_dim_value(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < 1.0 {
        return None;
    }
    Some(rounded as usize)
}

fn value_to_shift_vector(value: &Value) -> crate::BuiltinResult<Vec<isize>> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < isize::MIN as i64 || raw > isize::MAX as i64 {
                return Err(circshift_error("circshift: shift magnitude is too large"));
            }
            Ok(vec![raw as isize])
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(circshift_error(
                    "circshift: shift values must be finite numbers",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(circshift_error("circshift: shifts must be integers"));
            }
            if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
                return Err(circshift_error("circshift: shift magnitude is too large"));
            }
            Ok(vec![rounded as isize])
        }
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) && !tensor.data.is_empty() {
                return Err(circshift_error(
                    "circshift: shifts must be specified as a scalar or vector",
                ));
            }
            tensor
                .data
                .iter()
                .map(|val| numeric_to_isize(*val))
                .collect::<Result<Vec<_>, _>>()
        }
        Value::LogicalArray(array) => {
            if !is_vector_shape(&array.shape) && !array.data.is_empty() {
                return Err(circshift_error(
                    "circshift: shifts must be specified as a scalar or vector",
                ));
            }
            Ok(array
                .data
                .iter()
                .map(|&b| if b != 0 { 1 } else { 0 })
                .collect())
        }
        Value::Bool(flag) => Ok(vec![if *flag { 1 } else { 0 }]),
        Value::GpuTensor(_) => Err(circshift_error(
            "circshift: shift vector must reside on the host",
        )),
        Value::StringArray(_) | Value::CharArray(_) | Value::String(_) => {
            Err(circshift_error("circshift: shift values must be numeric"))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(circshift_error("circshift: shifts must be real integers"))
        }
        Value::Cell(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(circshift_error(
            "circshift: unsupported shift argument type",
        )),
    }
}

fn value_to_dims_vector(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 1 {
                return Err(circshift_error("circshift: dimensions must be >= 1"));
            }
            Ok(vec![raw as usize])
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(circshift_error(
                    "circshift: dimensions must be finite integers",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(circshift_error("circshift: dimensions must be integers"));
            }
            if rounded < 1.0 {
                return Err(circshift_error("circshift: dimensions must be >= 1"));
            }
            Ok(vec![rounded as usize])
        }
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) && !tensor.data.is_empty() {
                return Err(circshift_error(
                    "circshift: dimension vectors must be row or column vectors",
                ));
            }
            let mut dims = Vec::with_capacity(tensor.data.len());
            for &val in &tensor.data {
                if !val.is_finite() {
                    return Err(circshift_error(
                        "circshift: dimensions must be finite integers",
                    ));
                }
                let rounded = val.round();
                if (rounded - val).abs() > f64::EPSILON {
                    return Err(circshift_error("circshift: dimensions must be integers"));
                }
                if rounded < 1.0 {
                    return Err(circshift_error("circshift: dimensions must be >= 1"));
                }
                dims.push(rounded as usize);
            }
            Ok(dims)
        }
        Value::LogicalArray(array) => {
            if !is_vector_shape(&array.shape) && !array.data.is_empty() {
                return Err(circshift_error(
                    "circshift: dimension vectors must be row or column vectors",
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
        Value::Bool(flag) => {
            if *flag {
                Ok(vec![1])
            } else {
                Err(circshift_error("circshift: dimension indices must be >= 1"))
            }
        }
        Value::GpuTensor(_) => Err(circshift_error(
            "circshift: dimension vector must reside on the host",
        )),
        Value::StringArray(_) | Value::CharArray(_) | Value::String(_) => Err(circshift_error(
            "circshift: dimension indices must be numeric",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(circshift_error(
            "circshift: dimensions must be real integers",
        )),
        Value::Cell(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(circshift_error(
            "circshift: unsupported dimension argument type",
        )),
    }
}

fn numeric_to_isize(value: f64) -> crate::BuiltinResult<isize> {
    if !value.is_finite() {
        return Err(circshift_error("circshift: shift values must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(circshift_error("circshift: shift values must be integers"));
    }
    if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
        return Err(circshift_error("circshift: shift magnitude is too large"));
    }
    Ok(rounded as isize)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}

#[derive(Debug, Clone)]
struct ShiftPlan {
    ext_shape: Vec<usize>,
    positive: Vec<usize>,
    provider: Vec<isize>,
}

impl ShiftPlan {
    fn is_noop(&self) -> bool {
        self.positive.iter().all(|&shift| shift == 0)
    }
}

fn build_shift_plan(
    shape: &[usize],
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<ShiftPlan> {
    if dims.len() != shifts.len() {
        return Err(circshift_error(
            "circshift: shift vector must match the number of dimensions",
        ));
    }
    let mut target_len = shape.len();
    if let Some(max_axis) = dims.iter().copied().max() {
        if max_axis + 1 > target_len {
            target_len = max_axis + 1;
        }
    }
    let mut ext_shape = shape.to_vec();
    if target_len > ext_shape.len() {
        ext_shape.resize(target_len, 1);
    }

    let mut positive = vec![0usize; ext_shape.len()];
    let mut provider = vec![0isize; ext_shape.len()];

    for (&axis, &shift) in dims.iter().zip(shifts.iter()) {
        if axis >= ext_shape.len() {
            return Err(circshift_error("circshift: dimension index out of range"));
        }
        provider[axis] = shift;
        let size = ext_shape[axis];
        if size == 0 || size == 1 {
            continue;
        }
        let size_isize = size as isize;
        let mut normalized = shift % size_isize;
        if normalized < 0 {
            normalized += size_isize;
        }
        positive[axis] = normalized as usize;
    }

    Ok(ShiftPlan {
        ext_shape,
        positive,
        provider,
    })
}

fn normalize_shift_amount(shift: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let len_isize = len as isize;
    let mut normalized = shift % len_isize;
    if normalized < 0 {
        normalized += len_isize;
    }
    normalized as usize
}

fn circshift_tensor(
    tensor: Tensor,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<Tensor> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    Tensor::new(rotated, ext_shape).map_err(|e| circshift_error(format!("circshift: {e}")))
}

fn circshift_complex_tensor(
    tensor: ComplexTensor,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<ComplexTensor> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape)
            .map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    ComplexTensor::new(rotated, ext_shape).map_err(|e| circshift_error(format!("circshift: {e}")))
}

fn circshift_logical_array(
    array: LogicalArray,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape)
            .map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    LogicalArray::new(rotated, ext_shape).map_err(|e| circshift_error(format!("circshift: {e}")))
}

fn circshift_string_array(
    array: StringArray,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<StringArray> {
    let StringArray { data, shape, .. } = array;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return StringArray::new(data, shape)
            .map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    StringArray::new(rotated, ext_shape).map_err(|e| circshift_error(format!("circshift: {e}")))
}

fn circshift_char_array(
    array: CharArray,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<Value> {
    let mut row_shift = 0isize;
    let mut col_shift = 0isize;
    for (&axis, &shift) in dims.iter().zip(shifts.iter()) {
        match axis {
            0 => row_shift = shift,
            1 => col_shift = shift,
            _ => {
                if shift != 0 {
                    return Err(circshift_error(
                        "circshift: character arrays only support dimensions 1 and 2",
                    ));
                }
            }
        }
    }
    let CharArray { data, rows, cols } = array;
    if data.is_empty() {
        return CharArray::new(data, rows, cols)
            .map(Value::CharArray)
            .map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    let row_shift = normalize_shift_amount(row_shift, rows);
    let col_shift = normalize_shift_amount(col_shift, cols);
    if row_shift == 0 && col_shift == 0 {
        return CharArray::new(data, rows, cols)
            .map(Value::CharArray)
            .map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    let mut out = vec!['\0'; data.len()];
    for row in 0..rows {
        for col in 0..cols {
            let src_row = if rows == 0 {
                0
            } else {
                (row + rows - row_shift) % rows
            };
            let src_col = if cols == 0 {
                0
            } else {
                (col + cols - col_shift) % cols
            };
            let dst_idx = row * cols + col;
            let src_idx = src_row * cols + src_col;
            out[dst_idx] = data[src_idx];
        }
    }
    CharArray::new(out, rows, cols)
        .map(Value::CharArray)
        .map_err(|e| circshift_error(format!("circshift: {e}")))
}

async fn circshift_gpu(
    handle: GpuTensorHandle,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, shifts)?;
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return circshift_gpu_fallback(handle, dims, shifts).await,
            }
        }
        if let Ok(out) = provider.circshift(&working, &plan.provider) {
            return Ok(Value::GpuTensor(out));
        }
    }

    circshift_gpu_fallback(handle, dims, shifts).await
}

async fn circshift_gpu_fallback(
    handle: GpuTensorHandle,
    dims: &[usize],
    shifts: &[isize],
) -> crate::BuiltinResult<Value> {
    let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let rotated = circshift_tensor(host_tensor, dims, shifts)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &rotated.data,
            shape: &rotated.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| circshift_error(format!("circshift: {e}")));
    }
    Ok(tensor::tensor_into_value(rotated))
}

fn circshift_generic<T: Clone>(
    data: &[T],
    shape: &[usize],
    shifts: &[usize],
) -> crate::BuiltinResult<Vec<T>> {
    if shape.len() != shifts.len() {
        return Err(circshift_error("circshift: internal shape mismatch"));
    }
    let total: usize = shape.iter().product();
    if total != data.len() {
        return Err(circshift_error(
            "circshift: shape does not match data length",
        ));
    }
    if total == 0 {
        return Ok(Vec::new());
    }
    if shifts.iter().all(|&s| s == 0) {
        return Ok(data.to_vec());
    }

    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }

    let mut result = vec![data[0].clone(); total];
    let mut coords = vec![0usize; shape.len()];

    for (dest_idx, slot) in result.iter_mut().enumerate() {
        let mut remainder = dest_idx;
        for axis in 0..shape.len() {
            let size = shape[axis];
            coords[axis] = if size == 0 { 0 } else { remainder % size };
            if size > 0 {
                remainder /= size;
            }
        }
        let mut src_idx = 0usize;
        for axis in 0..shape.len() {
            let size = shape[axis];
            if size == 0 {
                continue;
            }
            let stride = strides[axis];
            if size <= 1 || shifts[axis] == 0 {
                src_idx += coords[axis] * stride;
            } else {
                let shift = shifts[axis] % size;
                let src_coord = (coords[axis] + size - shift) % size;
                src_idx += src_coord * stride;
            }
        }
        *slot = data[src_idx].clone();
    }

    Ok(result)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn circshift_builtin(
        value: Value,
        shift: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::circshift_builtin(value, shift, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor};

    #[test]
    fn circshift_type_preserves_tensor_shape() {
        let out = preserve_array_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(2)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_vector_positive_shift() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Int(IntValue::I32(2)),
            Vec::new(),
        )
        .expect("circshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_matrix_negative_column_shift() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let shift_vec = Tensor::new(vec![0.0, -1.0], vec![1, 2]).unwrap();
        let result =
            circshift_builtin(Value::Tensor(tensor), Value::Tensor(shift_vec), Vec::new()).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_column_vector_shift() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let shift_vec = Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap();
        let expected = circshift_tensor(tensor.clone(), &[0, 1], &[1, -2]).expect("expected shift");
        let result =
            circshift_builtin(Value::Tensor(tensor), Value::Tensor(shift_vec), Vec::new()).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_with_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Int(IntValue::I32(-1)),
            vec![Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![2.0, 4.0, 1.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_dims_tensor_argument() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let expected =
            circshift_tensor(tensor.clone(), &[0, 1], &[1, -1]).expect("expected host shift");
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(shift),
            vec![Value::Tensor(dims)],
        )
        .expect("circshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_logical_array_supported() {
        let array = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let shift_vec = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let result = circshift_builtin(
            Value::LogicalArray(array),
            Value::Tensor(shift_vec),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_dims_logical_mask() {
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 2, 2]).unwrap();
        let mask = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let expected =
            circshift_tensor(tensor.clone(), &[0, 2], &[1, -1]).expect("expected host shift");
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(shift),
            vec![Value::LogicalArray(mask)],
        )
        .expect("circshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_string_array_rotation() {
        let array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let result = circshift_builtin(
            Value::StringArray(array),
            Value::Int(IntValue::I32(-1)),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(
                    out.data,
                    vec!["b".to_string(), "c".to_string(), "a".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_char_array_rows() {
        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let result = circshift_builtin(
            Value::CharArray(chars),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data, vec!['c', 'd', 'a', 'b']);
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_noop_preserves_shape() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let shift_vec = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let result = circshift_builtin(
            Value::Tensor(tensor.clone()),
            Value::Tensor(shift_vec),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_rejects_duplicate_dims() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
            vec![Value::Tensor(dims)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("dimension indices must be unique"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_rejects_non_integer_shift() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err =
            circshift_builtin(Value::Tensor(tensor), Value::Num(1.5), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("shifts must be integers"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_dimension_length_mismatch() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let dims = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(shift),
            vec![Value::Tensor(dims)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("must have the same length"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = circshift_builtin(
                Value::GpuTensor(handle),
                Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
                Vec::new(),
            )
            .expect("circshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![4.0, 2.0, 3.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn circshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let cpu = circshift_builtin(
            Value::Tensor(tensor.clone()),
            Value::Tensor(shift.clone()),
            Vec::new(),
        )
        .expect("cpu circshift");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload tensor");
        let gpu_value =
            circshift_builtin(Value::GpuTensor(handle), Value::Tensor(shift), Vec::new())
                .expect("gpu circshift");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");
        match cpu {
            Value::Tensor(expected) => {
                assert_eq!(expected.shape, gathered.shape);
                assert_eq!(expected.data, gathered.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
