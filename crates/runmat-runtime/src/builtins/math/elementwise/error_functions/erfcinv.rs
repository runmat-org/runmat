//! Real `erfcinv` execution for RunMat.
//!
//! `erfcinv` computes the inverse complementary error function for real inputs.
//! The implementation uses a monotonic bisection over `erfc` to keep tails stable
//! without depending on a platform-specific inverse special function.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, ERFCINV_ERROR_INTERNAL, ERFCINV_ERROR_INVALID_ARGUMENT,
    ERFCINV_ERROR_INVALID_INPUT, ERFCINV_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use crate::builtins::math::resident_real_unary as real_unary;

const BUILTIN_NAME: &str = "erfcinv";
const MAX_POSITIVE_RESULT: f64 = 32.0;
const BISECTION_STEPS: usize = 110;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::error_functions::erfcinv"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_erfcinv",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate erfcinv directly on real device buffers; unsupported providers fall back to host evaluation and re-upload when possible.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::error_functions::erfcinv"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes:
        "Fusion planner currently falls back to provider or host elementwise erfcinv evaluation.",
};

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn internal_error(detail: impl AsRef<str>) -> RuntimeError {
    error_with_detail(&ERFCINV_ERROR_INTERNAL, detail)
}

#[runtime_builtin(
    name = "erfcinv",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::error_functions::erfcinv"
)]
async fn erfcinv_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    if !rest.is_empty() {
        return Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_ARGUMENT,
            "erfcinv accepts exactly one input",
        ));
    }
    match value {
        Value::GpuTensor(handle) => erfcinv_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "complex inputs are not supported",
        )),
        Value::String(_) | Value::StringArray(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "expected real numeric input, got string",
        )),
        Value::SparseTensor(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "sparse inputs are not supported",
        )),
        Value::Bool(_) | Value::LogicalArray(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "logical inputs are not supported",
        )),
        Value::Int(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "integer-class inputs are not supported",
        )),
        Value::CharArray(_) => Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            "char inputs are not supported",
        )),
        other => erfcinv_real(other),
    }
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(error_with_detail(
            &ERFCINV_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

async fn erfcinv_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Err(reason) = real_unary::validate_input(&handle) {
        return Err(error_with_detail(
            &ERFCINV_ERROR_INVALID_INPUT,
            reason.detail(),
        ));
    }

    let provider = real_unary::exact_owner(&handle, || {
        internal_error("GPU provider unavailable for input")
    })?;
    match provider.unary_erfcinv(&handle).await {
        Ok(mut out) if real_unary::output_matches(&out, &handle, provider) => {
            real_unary::preserve_residency_intent(&mut out, &handle)?;
            return gpu_helpers::resident_gpu_value(out);
        }
        Ok(out) => {
            return Err(real_unary::reject_output(
                out,
                internal_error("provider unary_erfcinv returned malformed output"),
            ));
        }
        Err(err) if real_unary::hook_allows_host_fallback(&err) => {}
        Err(err) => {
            let mut error = internal_error(format!("provider unary_erfcinv failed: {err}"));
            error = crate::runtime_error::with_provider_execution_source(error, err.into());
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = erfcinv_tensor(tensor)?;

    let out = real_unary::restore_fallback(&result, &handle, provider).map_err(|err| {
        internal_error(format!(
            "failed to restore fallback result to input provider: {err}"
        ))
    })?;
    gpu_helpers::resident_gpu_value(out)
}

fn erfcinv_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&ERFCINV_ERROR_INVALID_INPUT, detail))?;
    erfcinv_tensor(tensor).map(erfcinv_tensor_into_value)
}

fn erfcinv_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = match tensor.into_numeric_storage().map_err(internal_error)? {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(erfcinv_scalar).collect::<Vec<_>>())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| erfcinv_scalar(f64::from(value)) as f32)
                .collect::<Vec<_>>(),
        ),
        NumericStorage::I8(_)
        | NumericStorage::I16(_)
        | NumericStorage::I32(_)
        | NumericStorage::I64(_)
        | NumericStorage::U8(_)
        | NumericStorage::U16(_)
        | NumericStorage::U32(_)
        | NumericStorage::U64(_) => {
            return Err(error_with_detail(
                &ERFCINV_ERROR_INVALID_INPUT,
                "integer-class tensors are not supported",
            ))
        }
    };
    Tensor::from_numeric_storage(storage, shape).map_err(internal_error)
}

fn erfcinv_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 {
        let Some(NumericScalar::F64(value)) = tensor.numeric_value_at(0) else {
            unreachable!("scalar double erfcinv result has F64 storage")
        };
        Value::Num(value)
    } else {
        Value::Tensor(tensor)
    }
}

pub(crate) fn erfcinv_scalar(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=2.0).contains(&value) {
        return f64::NAN;
    }
    if value == 0.0 {
        return f64::INFINITY;
    }
    if value == 2.0 {
        return f64::NEG_INFINITY;
    }
    if value == 1.0 {
        return 0.0;
    }
    if value > 1.0 {
        return -erfcinv_positive_tail(2.0 - value);
    }
    erfcinv_positive_tail(value)
}

fn erfcinv_positive_tail(target: f64) -> f64 {
    debug_assert!(target > 0.0 && target < 1.0);
    let mut lo = 0.0;
    let mut hi = 1.0;
    while hi < MAX_POSITIVE_RESULT && libm::erfc(hi) > target {
        lo = hi;
        hi *= 2.0;
    }
    if libm::erfc(hi) > target {
        return hi;
    }

    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        if libm::erfc(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
#[path = "erfcinv/tests.rs"]
mod tests;
