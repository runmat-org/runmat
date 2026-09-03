//! MATLAB-compatible `rem` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, REM_ERROR_INTERNAL, REM_ERROR_INVALID_INPUT, REM_ERROR_SIZE_MISMATCH,
    REM_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{binary as common_binary, gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{
    reject_integer_logical_operands, try_integer_remainder, IntegerRemainderOp,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::rem")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rem",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_rem",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Native integer providers may execute exact rem directly. Floating execution may fuse; the standalone runtime path uses validated owner-specific host fallback and restores compatible residency.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::rem")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rem",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let a = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let b = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("{a} - {b} * trunc({a} / {b})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion expands rem as trunc(a / b) followed by a - b * q; providers may override with specialised kernels.",
};

const BUILTIN_NAME: &str = "rem";

fn rem_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    rem_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn rem_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "rem",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::rounding::rem"
)]
async fn rem_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    super::unary::reject_excess_outputs(BUILTIN_NAME, &REM_ERROR_TOO_MANY_OUTPUTS)?;
    match crate::builtins::table::plan_binary(lhs, rhs)
        .map_err(|error| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, error))?
    {
        common_binary::BinaryInputPlan::Values(values) => {
            let (left, right) = *values;
            rem_non_tabular(left, right).await
        }
        common_binary::BinaryInputPlan::Structured(plan) => {
            let (source, variables) = plan.variables();
            let mut output = Vec::with_capacity(variables.len());
            for (name, left, right) in variables {
                output.push((name, rem_non_tabular(left, right).await?));
            }
            crate::builtins::table::finish_binary(&source, output)
                .map_err(|error| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, error.to_string()))
        }
    }
}

async fn rem_non_tabular(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if crate::builtins::duration::is_duration_object(&lhs)
        || crate::builtins::duration::is_duration_object(&rhs)
    {
        let lhs = gather_value(lhs).await?;
        let rhs = gather_value(rhs).await?;
        return match super::binary::plan_duration(lhs, rhs, BUILTIN_NAME)
            .map_err(|error| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, error))?
        {
            common_binary::BinaryInputPlan::Values(values) => {
                let (left, right) = *values;
                rem_host(left, right)
            }
            common_binary::BinaryInputPlan::Structured(plan) => {
                let result = compute_rem_real(&plan.left, &plan.right)?;
                super::binary::finish_duration(plan, result, BUILTIN_NAME)
                    .map_err(|error| rem_error_with_detail(&REM_ERROR_INTERNAL, error.to_string()))
            }
        };
    }
    if matches!(&lhs, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(&rhs, Value::Complex(_, _) | Value::ComplexTensor(_))
    {
        return Err(rem_error_with_detail(
            &REM_ERROR_INVALID_INPUT,
            "inputs must be real",
        ));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&lhs, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&rhs, BUILTIN_NAME)?;
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME)
        .map_err(|error| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, error))?;
    match (lhs, rhs) {
        (Value::GpuTensor(a), Value::GpuTensor(b)) => rem_gpu_pair(a, b).await,
        (Value::GpuTensor(a), other) => {
            let gathered = gpu_helpers::gather_tensor_async(&a).await?;
            rem_host(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(b)) => {
            let gathered = gpu_helpers::gather_tensor_async(&b).await?;
            rem_host(other, Value::Tensor(gathered))
        }
        (left, right) => rem_host(left, right),
    }
}

async fn gather_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle)
            .await
            .map(Value::Tensor),
        other => Ok(other),
    }
}

async fn rem_gpu_pair(a: GpuTensorHandle, b: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_binary_inputs(&a, &b)
        .map_err(|error| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, error))?;
    if runmat_accelerate_api::handle_integer_type(&a).is_some()
        && runmat_accelerate_api::handle_integer_type(&b).is_some()
        && common_binary::matching_physical_inputs(&a, &b)
    {
        match provider.elem_rem(&a, &b).await {
            Ok(output) => {
                let contract = gpu_helpers::BinaryGpuOutputContract {
                    shape: a.shape.clone(),
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&a),
                    integer: runmat_accelerate_api::handle_integer_type(&a),
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                };
                return common_binary::validate_resident_output(
                    provider, &a, &b, output, &contract,
                )
                .map_err(|error| rem_error_with_detail(&REM_ERROR_INTERNAL, error));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                return Err(rem_error_with_detail(
                    &REM_ERROR_INTERNAL,
                    error.to_string(),
                ));
            }
        }
    }
    let left = gpu_helpers::gather_tensor_async(&a).await?;
    let right = gpu_helpers::gather_tensor_async(&b).await?;
    let result = rem_host(Value::Tensor(left), Value::Tensor(right))?;
    gpu_helpers::restore_class_preserving_value(&a, result, BUILTIN_NAME)
}

fn rem_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = try_integer_remainder(&lhs, &rhs, IntegerRemainderOp::Rem, BUILTIN_NAME)
        .map_err(|error| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, error))?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_rem_value(&lhs, &rhs) {
        return Ok(result);
    }
    let left = value_into_real_tensor(lhs)?;
    let right = value_into_real_tensor(rhs)?;
    compute_rem_real(&left, &right)
}

fn compute_rem_real(a: &Tensor, b: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| rem_error_with_detail(&REM_ERROR_SIZE_MISMATCH, err))?;
    let dtype = if a.numeric_dtype() == NumericDType::F32 || b.numeric_dtype() == NumericDType::F32
    {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    if plan.is_empty() {
        let tensor = Tensor::new_with_dtype(Vec::new(), plan.output_shape().to_vec(), dtype)
            .map_err(|e| rem_error_with_detail(&REM_ERROR_INTERNAL, e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let tensor = if dtype == NumericDType::F32 {
        let mut result = vec![0.0f32; plan.len()];
        for (out_idx, idx_a, idx_b) in plan.iter() {
            result[out_idx] =
                rem_real_scalar_f32(tensor_value_f32(a, idx_a), tensor_value_f32(b, idx_b));
        }
        Tensor::from_f32(result, plan.output_shape().to_vec())
    } else {
        let mut result = vec![0.0f64; plan.len()];
        for (out_idx, idx_a, idx_b) in plan.iter() {
            result[out_idx] = rem_real_scalar(
                tensor::tensor_value_f64(a, idx_a),
                tensor::tensor_value_f64(b, idx_b),
            );
        }
        Tensor::new(result, plan.output_shape().to_vec())
    }
    .map_err(|e| rem_error_with_detail(&REM_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn tensor_value_f32(tensor: &Tensor, index: usize) -> f32 {
    tensor.as_f32_slice().map_or_else(
        || tensor::tensor_value_f64(tensor, index) as f32,
        |data| data[index],
    )
}

fn rem_real_scalar_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() || (b == 0.0) || (!a.is_finite() && b.is_finite()) {
        return f32::NAN;
    }
    if b.is_infinite() && a.is_finite() {
        return if a == -0.0 { 0.0 } else { a };
    }
    let quotient = (a / b).trunc();
    if !quotient.is_finite() && b.is_finite() {
        return f32::NAN;
    }
    let remainder = a - b * quotient;
    if remainder == -0.0 {
        0.0
    } else {
        remainder
    }
}

fn rem_real_scalar(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if b == 0.0 {
        return f64::NAN;
    }
    if !a.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    if b.is_infinite() && a.is_finite() {
        return normalize_zero(a);
    }
    let quotient = (a / b).trunc();
    if !quotient.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    let remainder = a - b * quotient;
    normalize_zero(remainder)
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Some(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_rem_value(lhs: &Value, rhs: &Value) -> Option<Value> {
    if matches!(lhs, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(rhs, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
    {
        return None;
    }
    Some(Value::Num(rem_real_scalar(
        scalar_real_value(lhs)?,
        scalar_real_value(rhs)?,
    )))
}

fn normalize_zero(value: f64) -> f64 {
    if value == -0.0 {
        0.0
    } else {
        value
    }
}

fn value_into_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(rem_error_with_detail(
            &REM_ERROR_INVALID_INPUT,
            "inputs must be real",
        )),
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            Tensor::new(data, vec![ca.rows, ca.cols])
                .map_err(|e| rem_error_with_detail(&REM_ERROR_INTERNAL, e))
        }
        Value::String(_) | Value::StringArray(_) => Err(rem_error_with_detail(
            &REM_ERROR_INVALID_INPUT,
            "expected numeric input, got string",
        )),
        Value::GpuTensor(_) => Err(rem_error_with_detail(
            &REM_ERROR_INTERNAL,
            "internal error converting GPU tensor",
        )),
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other)
                .map_err(|err| rem_error_with_detail(&REM_ERROR_INVALID_INPUT, err))?;
            Ok(tensor)
        }
    }
}

#[cfg(test)]
#[path = "rem/tests.rs"]
pub(crate) mod tests;
