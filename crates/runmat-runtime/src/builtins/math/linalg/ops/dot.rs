//! MATLAB-compatible `dot` builtin with GPU-aware semantics for RunMat.
//!
//! Implements inner products for real and complex inputs, including dimension-aware
//! reductions that match MathWorks MATLAB behaviour. GPU inputs are gathered when
//! necessary and the result is re-uploaded to the active provider when possible so
//! downstream consumers can remain device-resident.

#[cfg(test)]
use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};
use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::DOT_DESCRIPTOR;
use runmat_builtins::{
    DOT_ERROR_INTERNAL, DOT_ERROR_INVALID_ARGUMENT, DOT_ERROR_INVALID_INPUT,
    DOT_INTEGER_DATA_EXTENSION, DOT_LOGICAL_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::dot_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DOT_NAME: &str = "dot";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::dot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dot",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Reduction { name: "dot" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(1024),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes: "Dispatches to a provider-side dot implementation when available; otherwise gathers operands and re-uploads real outputs.",
};

fn dot_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    dot_error_with_message(error.message, error)
}

fn dot_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(DOT_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn dot_invalid_argument(message: impl Into<String>) -> RuntimeError {
    dot_error_with_message(message, &DOT_ERROR_INVALID_ARGUMENT)
}

fn dot_invalid_input(message: impl Into<String>) -> RuntimeError {
    dot_error_with_message(message, &DOT_ERROR_INVALID_INPUT)
}

fn dot_internal_error(message: impl Into<String>) -> RuntimeError {
    dot_error_with_message(message, &DOT_ERROR_INTERNAL)
}

#[path = "dot/kernel_error.rs"]
pub mod kernel_error;
mod geometry;
pub(in crate::builtins::math) use geometry::{dot_geometry, DotGeometry};
use kernel_error::{DotKernelError, DotKernelResult};

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Tensor(_) | Value::LogicalArray(_) => {
            let dim = tensor::dimension_from_value_async(value, DOT_NAME, false)
                .await
                .map_err(dot_invalid_argument)?;
            dim.ok_or_else(|| {
                dot_invalid_argument(format!(
                    "{DOT_NAME}: dimension must be numeric, got {value:?}"
                ))
            })
        }
        _ => Err(dot_invalid_argument(format!(
            "{DOT_NAME}: dimension must be numeric, got {value:?}"
        ))),
    }
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(DOT_NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(DOT_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::dot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Higher-level fusion currently delegates to dedicated dot kernels or host fallbacks.",
};

#[runtime_builtin(
    name = "dot",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::dot"
)]
async fn dot_builtin(
    lhs: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(dot_invalid_argument("dot: too many input arguments"));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&lhs, DOT_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&rhs, DOT_NAME)?;
    if is_typed_integer_value(&lhs) || is_typed_integer_value(&rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOT_INTEGER_DATA_EXTENSION,
            DOT_NAME,
        )?;
    }
    if is_logical_value(&lhs) || is_logical_value(&rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOT_LOGICAL_DATA_EXTENSION,
            DOT_NAME,
        )?;
    }
    let dim = match rest.first() {
        Some(value) => Some(parse_dimension_arg(value).await?),
        None => None,
    };

    if let Some(value) = try_provider_dot(&lhs, &rhs, dim).await? {
        return Ok(value);
    }

    let lhs_gpu = matches!(lhs, Value::GpuTensor(_));
    let rhs_gpu = matches!(rhs, Value::GpuTensor(_));
    let resident_anchor = match (&lhs, &rhs) {
        (Value::GpuTensor(handle), _) | (_, Value::GpuTensor(handle)) => Some(handle.clone()),
        _ => None,
    };

    let lhs_host = gather_if_needed_async(&lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs_host = gather_if_needed_async(&rhs)
        .await
        .map_err(map_control_flow)?;

    let has_complex = value_is_complex(&lhs_host) || value_is_complex(&rhs_host);

    let value = if has_complex {
        let lhs_complex = value_into_complex_tensor(lhs_host)?;
        let rhs_complex = value_into_complex_tensor(rhs_host)?;
        let result = dot_complex_tensor(&lhs_complex, &rhs_complex, dim)?;
        complex_tensor_into_value(result)
    } else {
        let lhs_tensor =
            tensor::value_into_tensor_for(DOT_NAME, lhs_host).map_err(dot_invalid_input)?;
        let rhs_tensor =
            tensor::value_into_tensor_for(DOT_NAME, rhs_host).map_err(dot_invalid_input)?;
        let result = dot_real_tensor(&lhs_tensor, &rhs_tensor, dim)?;
        tensor::tensor_into_value(result)
    };

    if lhs_gpu || rhs_gpu {
        let protected = [&lhs, &rhs]
            .into_iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            })
            .collect::<Vec<_>>();
        promote_result_to_gpu(value, resident_anchor.as_ref(), &protected)
    } else {
        Ok(value)
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

async fn try_provider_dot(
    lhs: &Value,
    rhs: &Value,
    dim: Option<usize>,
) -> BuiltinResult<Option<Value>> {
    let (Value::GpuTensor(lhs_handle), Value::GpuTensor(rhs_handle)) = (lhs, rhs) else {
        return Ok(None);
    };
    if is_typed_integer_value(lhs)
        || is_typed_integer_value(rhs)
        || is_logical_value(lhs)
        || is_logical_value(rhs)
        || runmat_accelerate_api::handle_storage(lhs_handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_storage(rhs_handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
        || lhs_handle.shape() != rhs_handle.shape()
        || lhs_handle.device_id() != rhs_handle.device_id()
    {
        return Ok(None);
    }
    let Some(provider) = resolved_actual_dot_owner(lhs_handle) else {
        return Ok(None);
    };
    let Some(rhs_owner) = resolved_actual_dot_owner(rhs_handle) else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(provider, rhs_owner) {
        return Ok(None);
    }
    match provider.dot(lhs_handle, rhs_handle, dim).await {
        Ok(handle) if valid_provider_dot_output(&handle, lhs_handle, rhs_handle, provider, dim) => {
            Ok(Some(Value::GpuTensor(handle)))
        }
        Ok(handle) => Err(reject_dot_output(handle)),
        Err(err) if gpu_helpers::provider_hook_allows_host_fallback(&err) => Ok(None),
        Err(err) => {
            let mut error = dot_internal_error(format!("dot: provider execution failed: {err}"));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error = kernel_error::with_provider_source(error, err.into());
            Err(error)
        }
    }
}

fn valid_provider_dot_output(
    output: &runmat_accelerate_api::GpuTensorHandle,
    lhs: &runmat_accelerate_api::GpuTensorHandle,
    rhs: &runmat_accelerate_api::GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    dim: Option<usize>,
) -> bool {
    let shape = canonical_shape(lhs.shape());
    let target_dim = dim.unwrap_or_else(|| default_dimension(&shape));
    let mut expected_shape = shape;
    if target_dim <= expected_shape.len() {
        expected_shape[target_dim - 1] = 1;
    }
    output.shape() == expected_shape
        && output.device_id() == lhs.device_id()
        && !gpu_handles_alias(output, lhs)
        && !gpu_handles_alias(output, rhs)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output)
            == requested_dot_precision_for_handles(lhs, rhs)
        && resolved_actual_dot_owner(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn requested_dot_precision_for_handles(
    lhs: &runmat_accelerate_api::GpuTensorHandle,
    rhs: &runmat_accelerate_api::GpuTensorHandle,
) -> Option<runmat_accelerate_api::ProviderPrecision> {
    match (
        runmat_accelerate_api::handle_precision(lhs),
        runmat_accelerate_api::handle_precision(rhs),
    ) {
        (
            Some(runmat_accelerate_api::ProviderPrecision::F32),
            Some(runmat_accelerate_api::ProviderPrecision::F32),
        ) => Some(runmat_accelerate_api::ProviderPrecision::F32),
        (Some(_), Some(_)) => Some(runmat_accelerate_api::ProviderPrecision::F64),
        _ => None,
    }
}

fn canonical_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn value_into_complex_tensor(value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(t) => Ok(t),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|e| dot_invalid_input(format!("{DOT_NAME}: {e}"))),
        Value::Tensor(t) => real_tensor_to_complex(&t),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| dot_invalid_input(format!("{DOT_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| dot_invalid_input(format!("{DOT_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| dot_invalid_input(format!("{DOT_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(dot_invalid_input)?;
            real_tensor_to_complex(&tensor)
        }
        other => Err(dot_invalid_input(format!(
            "{DOT_NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn real_tensor_to_complex(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let shape = canonical_shape_tensor(tensor);
    let values = tensor::tensor_values_f64_cow(tensor);
    let mut data = Vec::with_capacity(values.len());
    for &value in values.iter() {
        data.push((value, 0.0));
    }
    ComplexTensor::from_f64_values_with_dtype(data, shape, tensor.numeric_dtype())
        .map_err(|e| dot_internal_error(format!("{DOT_NAME}: {e}")))
}

fn dot_real_tensor(a: &Tensor, b: &Tensor, dim: Option<usize>) -> DotKernelResult<Tensor> {
    let left_shape = canonical_shape_tensor(a);
    let right_shape = canonical_shape_tensor(b);
    if dim.is_none() && is_vector_shape(&left_shape) && is_vector_shape(&right_shape) {
        if tensor::tensor_element_len(a) != tensor::tensor_element_len(b) {
            return Err(DotKernelError::invalid_input());
        }
    } else {
        ensure_same_size(a, b)?;
    }
    let plan = DotGeometry::new(&left_shape, &right_shape, dim)?;
    if plan.elementwise() {
        return elementwise_real_product(a, b);
    }
    let exact_integer_operands = a.integer_storage().is_some() && b.integer_storage().is_some();
    let floating_values = if exact_integer_operands {
        None
    } else {
        Some((
            tensor::tensor_values_f64_cow(a),
            tensor::tensor_values_f64_cow(b),
        ))
    };
    let mut output = Vec::with_capacity(plan.outputs());
    for row in 0..plan.outputs() {
        let mut acc = 0.;
        for axis in 0..plan.factor() {
            let index = plan
                .source(row, axis)
                .ok_or_else(|| DotKernelError::internal("dot source geometry overflow"))?;
            let product = if exact_integer_operands {
                exact_integer_product_as_f64(a.numeric_value_at(index), b.numeric_value_at(index))
                    .ok_or_else(|| DotKernelError::internal("dot: invalid integer storage"))?
            } else {
                let (left, right) = floating_values.as_ref().expect("floating dot operands");
                left[index] * right[index]
            };
            acc += product;
        }
        output.push(acc);
    }
    dot_real_output(
        output,
        plan.output_shape().to_vec(),
        dot_real_output_dtype(a, b),
    )
}

fn dot_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> DotKernelResult<ComplexTensor> {
    if dim.is_none()
        && is_vector_shape(&canonical_shape_complex(a))
        && is_vector_shape(&canonical_shape_complex(b))
    {
        if a.materialize_f64().len() != b.materialize_f64().len() {
            return Err(DotKernelError::invalid_input());
        }
        let mut output = (0.0, 0.0);
        for ((ar, ai), (br, bi)) in a.materialize_f64().iter().zip(&b.materialize_f64()) {
            output.0 += ar * br + ai * bi;
            output.1 += ar * bi - ai * br;
        }
        return ComplexTensor::from_f64_values_with_dtype(
            vec![output],
            vec![1, 1],
            dot_complex_output_dtype(a, b),
        )
        .map_err(|error| DotKernelError::internal(format!("{DOT_NAME}: {error}")));
    }
    ensure_same_size_complex(a, b)?;

    let shape = canonical_shape_complex(a);
    let target_dim = dim.unwrap_or_else(|| default_dimension(&shape));
    let dim_index = target_dim - 1;

    if dim_index >= shape.len() {
        return elementwise_complex_product(a, b);
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let mut output = vec![(0.0f64, 0.0f64); stride_before * stride_after];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let (ar, ai) = a.materialize_f64()[idx];
                let (br, bi) = b.materialize_f64()[idx];
                let real = ar * br + ai * bi;
                let imag = ar * bi - ai * br;
                acc_re += real;
                acc_im += imag;
            }
            let out_idx = after * stride_before + before;
            output[out_idx] = (acc_re, acc_im);
        }
    }

    let mut out_shape = shape.clone();
    out_shape[dim_index] = 1;
    ComplexTensor::from_f64_values_with_dtype(output, out_shape, dot_complex_output_dtype(a, b))
        .map_err(|e| DotKernelError::internal(format!("{DOT_NAME}: {e}")))
}

pub fn dot_host_real_for_provider(
    a: &Tensor,
    b: &Tensor,
    dim: Option<usize>,
) -> DotKernelResult<Tensor> {
    dot_real_tensor(a, b, dim)
}

pub fn dot_host_complex_for_provider(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> DotKernelResult<ComplexTensor> {
    dot_complex_tensor(a, b, dim)
}

fn elementwise_real_product(a: &Tensor, b: &Tensor) -> DotKernelResult<Tensor> {
    let mut data = Vec::with_capacity(tensor::tensor_element_len(a));
    let exact_integer_operands = a.integer_storage().is_some() && b.integer_storage().is_some();
    if exact_integer_operands {
        for index in 0..tensor::tensor_element_len(a) {
            data.push(
                exact_integer_product_as_f64(a.numeric_value_at(index), b.numeric_value_at(index))
                    .ok_or_else(|| DotKernelError::internal("dot: invalid integer storage"))?,
            );
        }
    } else {
        let a_values = tensor::tensor_values_f64_cow(a);
        let b_values = tensor::tensor_values_f64_cow(b);
        for index in 0..a_values.len() {
            data.push(a_values[index] * b_values[index]);
        }
    }
    let shape = canonical_shape_tensor(a);
    dot_real_output(data, shape, dot_real_output_dtype(a, b))
}

fn dot_real_output_dtype(a: &Tensor, b: &Tensor) -> NumericDType {
    if a.numeric_dtype() == NumericDType::F32 && b.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn dot_complex_output_dtype(a: &ComplexTensor, b: &ComplexTensor) -> NumericDType {
    if a.numeric_dtype() == NumericDType::F32 && b.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn dot_real_output(
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
) -> DotKernelResult<Tensor> {
    match dtype {
        NumericDType::F32 => {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
        }
        _ => Tensor::new(data, shape),
    }
    .map_err(|error| DotKernelError::internal(format!("{DOT_NAME}: {error}")))
}

fn exact_integer_product_as_f64(
    lhs: Option<NumericScalar>,
    rhs: Option<NumericScalar>,
) -> Option<f64> {
    let (lhs_negative, lhs_magnitude) = integer_sign_magnitude(lhs?)?;
    let (rhs_negative, rhs_magnitude) = integer_sign_magnitude(rhs?)?;
    let magnitude = lhs_magnitude.checked_mul(rhs_magnitude)?;
    let value = magnitude as f64;
    Some(if lhs_negative ^ rhs_negative {
        -value
    } else {
        value
    })
}

fn integer_sign_magnitude(value: NumericScalar) -> Option<(bool, u128)> {
    let signed = match value {
        NumericScalar::I8(value) => i128::from(value),
        NumericScalar::I16(value) => i128::from(value),
        NumericScalar::I32(value) => i128::from(value),
        NumericScalar::I64(value) => i128::from(value),
        NumericScalar::U8(value) => return Some((false, u128::from(value))),
        NumericScalar::U16(value) => return Some((false, u128::from(value))),
        NumericScalar::U32(value) => return Some((false, u128::from(value))),
        NumericScalar::U64(value) => return Some((false, u128::from(value))),
        NumericScalar::F32(_) | NumericScalar::F64(_) => return None,
    };
    Some((signed.is_negative(), signed.unsigned_abs()))
}

fn elementwise_complex_product(
    a: &ComplexTensor,
    b: &ComplexTensor,
) -> DotKernelResult<ComplexTensor> {
    let mut data = Vec::with_capacity(a.materialize_f64().len());
    for ((ar, ai), (br, bi)) in a.materialize_f64().iter().zip(&b.materialize_f64()) {
        let real = ar * br + ai * bi;
        let imag = ar * bi - ai * br;
        data.push((real, imag));
    }
    let shape = canonical_shape_complex(a);
    ComplexTensor::from_f64_values_with_dtype(data, shape, dot_complex_output_dtype(a, b))
        .map_err(|e| DotKernelError::internal(format!("{DOT_NAME}: {e}")))
}

fn ensure_same_size(a: &Tensor, b: &Tensor) -> DotKernelResult<()> {
    if tensor::tensor_element_len(a) != tensor::tensor_element_len(b) {
        return Err(DotKernelError::invalid_input());
    }
    if canonical_shape_tensor(a) != canonical_shape_tensor(b) {
        return Err(DotKernelError::invalid_input());
    }
    Ok(())
}

fn ensure_same_size_complex(a: &ComplexTensor, b: &ComplexTensor) -> DotKernelResult<()> {
    if a.materialize_f64().len() != b.materialize_f64().len() {
        return Err(DotKernelError::invalid_input());
    }
    if canonical_shape_complex(a) != canonical_shape_complex(b) {
        return Err(DotKernelError::invalid_input());
    }
    Ok(())
}

fn canonical_shape_tensor(t: &Tensor) -> Vec<usize> {
    if t.shape().is_empty() {
        // Dot interprets a rank-zero tensor as an empty 0-by-0 matrix rather
        // than using Tensor's general scalar projection.
        vec![0, 0]
    } else {
        t.shape().to_vec()
    }
}

fn canonical_shape_complex(t: &ComplexTensor) -> Vec<usize> {
    if t.shape().is_empty() {
        vec![t.rows(), t.cols()]
    } else {
        t.shape().to_vec()
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    (rows == 1 || cols == 1) && shape.iter().skip(2).all(|dimension| *dimension == 1)
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim))
}

fn promote_result_to_gpu(
    value: Value,
    anchor: Option<&runmat_accelerate_api::GpuTensorHandle>,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    match anchor {
        Some(anchor) => gpu_helpers::restore_class_preserving_value_protecting(
            anchor, value, DOT_NAME, protected,
        ),
        None => Ok(value),
    }
}

fn resolved_actual_dot_owner(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    gpu_helpers::exact_provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id())
}

fn gpu_handles_alias(
    lhs: &runmat_accelerate_api::GpuTensorHandle,
    rhs: &runmat_accelerate_api::GpuTensorHandle,
) -> bool {
    lhs.shares_allocation_with(rhs)
}

fn reject_dot_output(handle: runmat_accelerate_api::GpuTensorHandle) -> RuntimeError {
    let mut error = dot_internal_error("dot: provider returned malformed or aliased output");
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    gpu_helpers::reject_gpu_output(handle, error)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{LiteralValue, ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage, LogicalArray};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_row_vectors() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Num(result) => assert_eq!(result, 32.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn dot_type_reduces_first_dimension() {
        let out = dot_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(2)])
            }
        );
    }

    #[test]
    fn dot_type_vector_with_dim_returns_scalar() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
        ]);
        let out = dot_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(4)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(4)]),
                },
                Type::Int,
            ],
            &ctx,
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn dot_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"C = dot(A, B)"));
        assert!(labels.contains(&"C = dot(A, B, dim)"));
    }

    #[test]
    fn dot_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = DOT_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.DOT.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.DOT.INVALID_INPUT"));
        assert!(codes.contains(&"RM.DOT.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_column_vectors() {
        let lhs = Tensor::new(vec![1.0, 3.0, 5.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Num(result) => assert_eq!(result, 44.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn dot_preserves_single_class_for_real_and_complex_host_inputs() {
        let lhs = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::from_f32(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let Value::Tensor(real) =
            dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
        else {
            panic!("single scalar remains a typed tensor")
        };
        assert_eq!(real.numeric_dtype(), NumericDType::F32);
        assert_eq!(real.materialize_f64(), vec![32.0]);

        let lhs = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::from_f32(vec![(3.0, 0.0), (4.0, 0.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(complex) = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .unwrap() else {
            panic!("single complex scalar remains a typed complex tensor")
        };
        assert_eq!(complex.numeric_dtype(), NumericDType::F32);
        assert_eq!(complex.materialize_f64(), vec![(11.0, 1.0)]);
    }

    #[test]
    fn dot_validates_requested_precision_and_hostile_result_metadata() {
        test_support::with_test_provider(|provider| {
            let upload_single = |data: &[f32], shape: &[usize]| {
                runmat_accelerate_api::upload_numeric_protecting(
                    provider,
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(data),
                        shape,
                        storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    },
                    &[],
                )
                .unwrap()
            };
            let lhs_single = upload_single(&[1.0; 6], &[2, 3]);
            let rhs_single = upload_single(&[2.0; 6], &[2, 3]);
            let single_output = upload_single(&[4.0; 3], &[1, 3]);
            assert!(valid_provider_dot_output(
                &single_output,
                &lhs_single,
                &rhs_single,
                provider,
                Some(1)
            ));
            let double_output = provider
                .upload(&HostTensorView {
                    data: &[4.0; 3],
                    shape: &[1, 3],
                })
                .unwrap();
            assert!(!valid_provider_dot_output(
                &double_output,
                &lhs_single,
                &rhs_single,
                provider,
                Some(1)
            ));

            let lhs = provider
                .upload(&HostTensorView {
                    data: &[1.0; 6],
                    shape: &[2, 3],
                })
                .unwrap();
            let rhs = provider
                .upload(&HostTensorView {
                    data: &[2.0; 6],
                    shape: &[2, 3],
                })
                .unwrap();
            assert!(valid_provider_dot_output(
                &double_output,
                &lhs,
                &rhs,
                provider,
                Some(1)
            ));
            let wrong_shape = double_output
                .with_view(double_output.view().clone().with_shape(vec![3, 1]))
                .unwrap();
            let complex = gpu_helpers::upload_complex_tensor_protecting(
                provider,
                &ComplexTensor::new(vec![(1.0, 2.0); 3], vec![1, 3]).unwrap(),
                &[&lhs, &rhs],
            )
            .unwrap();
            let integer = runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::I16(&[1; 3]),
                    shape: &[1, 3],
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                },
                &[&lhs, &rhs],
            )
            .unwrap();
            let logical = runmat_accelerate_api::upload_logical_protecting(
                provider,
                &runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1; 3],
                    shape: &[1, 3],
                },
                &[&lhs, &rhs],
            )
            .unwrap();
            for rejected in [wrong_shape, complex, integer, logical] {
                assert!(!valid_provider_dot_output(
                    &rejected,
                    &lhs,
                    &rhs,
                    provider,
                    Some(1)
                ));
                let error = reject_dot_output(rejected);
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            }
            let alias = lhs.with_view(lhs.view().clone()).unwrap();
            assert!(!valid_provider_dot_output(
                &alias,
                &lhs,
                &rhs,
                provider,
                Some(1)
            ));
            assert!(alias.shares_allocation_with(&lhs));
            let _error = reject_dot_output(alias);
            lhs.allocation_lease().validate_live().unwrap();
            rhs.allocation_lease().validate_live().unwrap();

            let foreign_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
            let foreign = foreign_owner
                .allocate_zeroed(
                    vec![1, 3],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            assert!(!valid_provider_dot_output(
                &foreign.handle,
                &lhs,
                &rhs,
                provider,
                Some(1)
            ));
            assert!(resolved_actual_dot_owner(&foreign.handle).is_none());
            let _error = reject_dot_output(foreign.handle);
            assert_eq!(foreign.probe.retirements(), 1);
            assert_eq!(
                block_on(provider.download(&lhs)).unwrap().data,
                vec![1.0; 6]
            );
        });
    }

    #[test]
    fn dot_vectors_need_equal_length_not_equal_orientation() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
        assert_eq!(
            dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap(),
            Value::Num(32.0)
        );

        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(3.0, 0.0), (4.0, 0.0)], vec![2, 1]).unwrap();
        assert_eq!(
            dot_builtin(
                Value::ComplexTensor(lhs),
                Value::ComplexTensor(rhs),
                Vec::new(),
            )
            .unwrap(),
            Value::Complex(11.0, 1.0)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn dot_preserves_rank_zero_tensor_as_zero_by_zero() {
        let lhs = Tensor::new(vec![2.0], Vec::new()).expect("rank-zero lhs");
        let rhs = Tensor::new(vec![3.0], Vec::new()).expect("rank-zero rhs");

        assert!(lhs.shape().is_empty());
        assert_eq!(lhs.rows(), 1, "the general Tensor projection is scalar");
        assert_eq!(canonical_shape_tensor(&lhs), [0, 0]);

        let Value::Tensor(result) =
            dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot")
        else {
            panic!("rank-zero tensors follow the tensor dot path");
        };
        assert_eq!(result.shape(), [1, 0]);
        assert!(result.materialize_f64().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_with_dimension_argument() {
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], vec![2, 3]).unwrap();
        let cols = dot_builtin(
            Value::Tensor(lhs.clone()),
            Value::Tensor(rhs.clone()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("dot");
        match cols {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 3]);
                assert_eq!(t.materialize_f64(), vec![18.0, 20.0, 18.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let rows = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("dot");
        match rows {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [2, 1]);
                assert_eq!(t.materialize_f64(), vec![28.0, 28.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn dot_reads_typed_integer_tensors_and_dimension_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 2, 5, 3, 6]), vec![2, 3])
            .expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U16(vec![6, 3, 5, 2, 4, 1]), vec![2, 3])
            .expect("rhs");
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("dim");

        let value = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Tensor(dim)],
        )
        .expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 3]);
                assert_eq!(t.materialize_f64(), vec![18.0, 20.0, 18.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn dot_integer_extension_multiplies_wide_values_before_float_conversion() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = (1_u64 << 53) + 1;
        let lhs = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let rhs = Tensor::new_integer(IntegerStorage::U8(vec![3]), vec![1, 1]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap();
        let expected = (u128::from(wide) * 3) as f64;
        let preconverted = (wide as f64) * 3.0;
        assert_ne!(
            expected, preconverted,
            "test must straddle the f64 boundary"
        );
        assert_eq!(value, Value::Num(expected));
    }

    #[test]
    fn dot_integer_reduction_and_out_of_rank_paths_use_authoritative_storage() {
        let wide = (1_u64 << 53) + 1;
        let lhs = Tensor::new_integer(
            IntegerStorage::U64(vec![wide, wide + 2, wide + 4, wide + 6]),
            vec![2, 2],
        )
        .unwrap();
        let rhs = Tensor::new_integer(IntegerStorage::U8(vec![3, 5, 7, 11]), vec![2, 2]).unwrap();

        let exact_products = vec![
            (u128::from(wide) * 3) as f64,
            (u128::from(wide + 2) * 5) as f64,
            (u128::from(wide + 4) * 7) as f64,
            (u128::from(wide + 6) * 11) as f64,
        ];
        let preconverted_products = vec![
            (wide as f64) * 3.0,
            ((wide + 2) as f64) * 5.0,
            ((wide + 4) as f64) * 7.0,
            ((wide + 6) as f64) * 11.0,
        ];
        assert_ne!(exact_products, preconverted_products);

        let elementwise = dot_real_tensor(&lhs, &rhs, Some(3)).unwrap();
        assert_eq!(elementwise.shape(), [2, 2]);
        assert_eq!(elementwise.materialize_f64(), exact_products);

        let reduced = dot_real_tensor(&lhs, &rhs, Some(1)).unwrap();
        assert_eq!(reduced.shape(), [1, 2]);
        assert_eq!(
            reduced.materialize_f64(),
            vec![
                exact_products[0] + exact_products[1],
                exact_products[2] + exact_products[3],
            ]
        );
    }

    #[test]
    fn dot_accepts_every_integer_class_for_documented_dimension_control() {
        let lhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let rhs = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap());
        for storage in [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ] {
            let dim = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            assert_eq!(
                dot_builtin(lhs.clone(), rhs.clone(), vec![dim]).unwrap(),
                Value::Num(11.0)
            );
        }
    }

    #[test]
    fn dot_integer_and_logical_data_extensions_are_gated_before_execution() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_error = dot_builtin(
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U8(3)),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            integer_error.identifier(),
            Some("RunMat:compatibility:DotIntegerDataExtension")
        );

        let logical_error =
            dot_builtin(Value::Bool(true), Value::Bool(false), Vec::new()).unwrap_err();
        assert_eq!(
            logical_error.identifier(),
            Some("RunMat:compatibility:DotLogicalDataExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_with_dimension() {
        let lhs = ComplexTensor::new(
            vec![(1.0, 1.0), (3.0, -2.0), (2.0, -3.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let rhs = ComplexTensor::new(
            vec![(2.0, -1.0), (1.0, 4.0), (-1.0, 2.0), (3.0, 5.0)],
            vec![2, 2],
        )
        .unwrap();
        let value = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("dot");
        match value {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                let expected = [(-4.0, 11.0), (4.0, 21.0)];
                for (idx, (got, exp)) in t.materialize_f64().iter().zip(expected.iter()).enumerate()
                {
                    assert!(
                        (got.0 - exp.0).abs() < 1e-12,
                        "real mismatch at {idx}: got {}, expected {}",
                        got.0,
                        exp.0
                    );
                    assert!(
                        (got.1 - exp.1).abs() < 1e-12,
                        "imag mismatch at {idx}: got {}, expected {}",
                        got.1,
                        exp.1
                    );
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_uses_conjugate_first_argument() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -3.0), (-1.0, 5.0)], vec![1, 2]).unwrap();
        let value = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("dot");
        match value {
            Value::Complex(re, im) => {
                assert!((re + 27.0).abs() < 1e-12, "expected real -27, got {re}");
                assert!((im - 4.0).abs() < 1e-12, "expected imag 4, got {im}");
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_and_real_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let value =
            dot_builtin(Value::ComplexTensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Complex(re, im) => {
                assert!((re - 11.0).abs() < 1e-12, "expected real 11, got {re}");
                assert!((im - 1.0).abs() < 1e-12, "expected imag 1, got {im}");
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_empty_reduction_returns_zero() {
        let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 0.0]);
                assert_eq!(t.shape(), [1, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_mismatched_shapes_error() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect_err("dot"),
        );
        assert_eq!(err.identifier(), DOT_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("A and B must be the same size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_zero_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            dot_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect_err("expected dimension error"),
        );
        assert_eq!(err.identifier(), DOT_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_non_integer_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            dot_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Num(1.5)],
            )
            .expect_err("expected integer dimension error"),
        );
        assert_eq!(err.identifier(), DOT_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("dimension must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_promotes_logical_inputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = dot_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("dot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 7.0]);
                assert_eq!(t.shape(), [1, 2]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![1, 4]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: lhs.shape(),
            };
            let view_rhs = HostTensorView {
                data: &rhs.materialize_f64(),
                shape: rhs.shape(),
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
            let value = dot_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::GpuTensor(gpu_rhs),
                Vec::new(),
            )
            .expect("dot");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape(), [1, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![20.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_mixed_gpu_and_host_returns_gpu() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![1, 4]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: lhs.shape(),
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let value = dot_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::Tensor(rhs.clone()),
                Vec::new(),
            )
            .expect("dot");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather result");
                    assert_eq!(gathered.shape(), [1, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![20.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_exceeds_rank_returns_product() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let value = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Num(3.0)],
        )
        .expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                assert_eq!(t.materialize_f64(), vec![3.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires an actual WGPU adapter; run the explicit hardware gate"]
    fn dot_wgpu_matches_cpu() {
        use runmat_accelerate_api::AccelProvider;

        let _guard = test_support::accel_test_lock();
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("requested vector-product hardware gate requires an actual WGPU adapter");
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![6.0, 3.0, 5.0, 1.0], vec![2, 2]).unwrap();
        let cpu = dot_real_tensor(&lhs, &rhs, Some(1)).expect("cpu dot");
        let view_lhs = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: lhs.shape(),
        };
        let view_rhs = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: rhs.shape(),
        };
        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
        let gpu_value = dot_builtin(
            Value::GpuTensor(gpu_lhs),
            Value::GpuTensor(gpu_rhs),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("gpu dot");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape(), cpu.shape());
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }

    fn dot_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::dot_builtin(
            lhs,
            rhs,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
}
