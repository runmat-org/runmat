//! MATLAB-compatible `atan2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, ATAN2_CHARACTER_INPUT_EXTENSION, ATAN2_ERROR_COMPLEX_UNSUPPORTED,
    ATAN2_ERROR_INTERNAL, ATAN2_ERROR_INVALID_INPUT, ATAN2_ERROR_SIZE_MISMATCH,
    ATAN2_ERROR_TOO_MANY_OUTPUTS, ATAN2_INTEGER_INPUT_EXTENSION, ATAN2_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::NumericStorage;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{
    binary as common_binary, broadcast::BroadcastPlan, gpu_helpers, tensor,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atan2";

mod admission;
pub(in crate::builtins::math::trigonometry) mod binary_angle;
mod preparation;
mod tabular;

#[cfg(test)]
#[path = "atan2/admission_tests.rs"]
mod admission_tests;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atan2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_atan2",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can implement elem_atan2 to keep the computation on device; the runtime gathers operands to the host when the hook is unavailable or broadcasting is required.",
};

fn atan2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn atan2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atan2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let y = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let x = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            match ctx.scalar_ty {
                ScalarType::F32 => writer.write_formatted(format_args!("select(atan2({y}, {x}), select({y}, 0.0, bitcast<u32>({x}) == 0x80000000u), ({y} == 0.0) && ({x} == 0.0))")),
                ScalarType::F64 => writer.write_formatted(format_args!("select(atan2({y}, {x}), select({y}, 0.0, bitcast<u64>({x}) == 0x8000000000000000u), ({y} == 0.0) && ({x} == 0.0))")),
                other => Err(FusionError::UnsupportedPrecision(other)),
            }
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits MATLAB-compatible atan2(y, x), returning positive zero for either signed-zero numerator when the denominator is negative zero while preserving the numerator sign for a positive-zero denominator; providers may override via elem_atan2 for standalone execution.",
};

#[runtime_builtin(
    name = "atan2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::trigonometry::atan2"
)]
async fn atan2_builtin(
    runtime: Option<RuntimeContext>,
    y: Value,
    x: Value,
) -> BuiltinResult<RootedValue> {
    // Planning can row-select right-hand fields and then drop the original
    // Object. Both original operands must be rooted before that begins.
    let _inputs = RootedValueLease::for_values([&y, &x].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    reject_excess_outputs()?;
    match crate::builtins::table::plan_binary(y, x)
        .map_err(|error| atan2_error_with_detail(&ATAN2_ERROR_INVALID_INPUT, error))?
    {
        common_binary::BinaryInputPlan::Values(values) => {
            let (y, x) = *values;
            preparation::execute(runtime.as_ref(), y, x).await
        }
        common_binary::BinaryInputPlan::Structured(plan) => {
            let crate::builtins::table::TabularBinaryPlan {
                source,
                variables,
                right_values,
            } = plan;
            tabular::execute(source, variables, right_values).await
        }
    }
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(atan2_error(&ATAN2_ERROR_TOO_MANY_OUTPUTS));
    }
    Ok(())
}

async fn atan2_admitted(y: Value, x: Value, protected: &[GpuTensorHandle]) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_values(&[&y, &x], true, || {
        atan2_error_with_detail(
            &ATAN2_ERROR_INVALID_INPUT,
            "resident inputs require one exact owning provider",
        )
    })?;
    match (y, x) {
        (Value::GpuTensor(yh), Value::GpuTensor(xh)) => atan2_gpu_pair(yh, xh, protected).await,
        (Value::GpuTensor(yh), other) => {
            let provider = gpu_helpers::require_resident_compute_owner(&yh, || {
                atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, "GPU input has no owning provider")
            })?;
            let gathered = gpu_helpers::gather_tensor_async(&yh).await?;
            let output = atan2_host(Value::Tensor(gathered), other)?;
            crate::builtins::common::provider_restore::upload_value_like_protected(
                provider,
                output,
                BUILTIN_NAME,
                &yh,
                protected,
            )
        }
        (other, Value::GpuTensor(xh)) => {
            let provider = gpu_helpers::require_resident_compute_owner(&xh, || {
                atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, "GPU input has no owning provider")
            })?;
            let gathered = gpu_helpers::gather_tensor_async(&xh).await?;
            let output = atan2_host(other, Value::Tensor(gathered))?;
            crate::builtins::common::provider_restore::upload_value_like_protected(
                provider,
                output,
                BUILTIN_NAME,
                &xh,
                protected,
            )
        }
        (lhs, rhs) => atan2_host(lhs, rhs),
    }
}

async fn atan2_gpu_pair(
    y: GpuTensorHandle,
    x: GpuTensorHandle,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let owner = gpu_helpers::exact_provider_for_binary_inputs(&y, &x)
        .map_err(|error| atan2_error_with_detail(&ATAN2_ERROR_INVALID_INPUT, error))?;
    let nonfloating = |handle: &GpuTensorHandle| {
        runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
    };
    if common_binary::matching_physical_inputs(&y, &x)
        && runmat_accelerate_api::handle_storage(&y) == GpuTensorStorage::Real
        && !nonfloating(&y)
        && !nonfloating(&x)
    {
        match owner.elem_atan2(&y, &x).await {
            Ok(output) => {
                if protected
                    .iter()
                    .any(|input| gpu_helpers::same_gpu_handle(input, &output))
                {
                    let mut error = atan2_error_with_detail(
                        &ATAN2_ERROR_INTERNAL,
                        "provider aliased a borrowed coordinate operand",
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(output, error));
                }
                let contract = gpu_helpers::BinaryGpuOutputContract {
                    shape: y.shape().to_vec(),
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&y),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                };
                return common_binary::validate_resident_output(owner, &y, &x, output, &contract)
                    .map_err(|source| {
                        let mut error =
                            atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, source.message());
                        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        error.source = Some(Box::new(source));
                        error
                    });
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => {
                return Err(atan2_error_with_detail(
                    &ATAN2_ERROR_INTERNAL,
                    error.to_string(),
                ));
            }
        }
    }
    let host_y = gpu_helpers::gather_tensor_async(&y).await?;
    let host_x = gpu_helpers::gather_tensor_async(&x).await?;
    let output = atan2_host(Value::Tensor(host_y), Value::Tensor(host_x))?;
    let prototype = if runmat_accelerate_api::handle_is_explicit(&x)
        && !runmat_accelerate_api::handle_is_explicit(&y)
    {
        &x
    } else {
        &y
    };
    let mut protected = protected.to_vec();
    protected.extend([y.clone(), x.clone()]);
    crate::builtins::common::provider_restore::upload_value_like_protected(
        owner,
        output,
        BUILTIN_NAME,
        prototype,
        &protected,
    )
}

fn atan2_host(y: Value, x: Value) -> BuiltinResult<Value> {
    let tensor_y = value_into_atan2_tensor(y)?;
    let tensor_x = value_into_atan2_tensor(x)?;
    compute_atan2_tensor(&tensor_y, &tensor_x)
}

fn compute_atan2_tensor(y: &Tensor, x: &Tensor) -> BuiltinResult<Value> {
    let result =
        binary_angle::compute(y, x, binary_angle::Unit::Radians).map_err(|cause| match cause {
            binary_angle::ComputeError::SizeMismatch(detail) => {
                atan2_error_with_detail(&ATAN2_ERROR_SIZE_MISMATCH, detail)
            }
            binary_angle::ComputeError::Internal(detail) => {
                atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, detail)
            }
        })?;
    Ok(tensor::tensor_into_value(result))
}

pub(super) fn matlab_atan2_f64(y: f64, x: f64) -> f64 {
    if y == 0.0 && x == 0.0 && x.is_sign_negative() {
        0.0
    } else {
        y.atan2(x)
    }
}

pub(super) fn matlab_atan2_f32(y: f32, x: f32) -> f32 {
    if y == 0.0 && x == 0.0 && x.is_sign_negative() {
        0.0
    } else {
        y.atan2(x)
    }
}

fn ensure_atan2_input_extensions(value: &Value) -> BuiltinResult<()> {
    super::inverse_helpers::ensure_input_extensions(
        value,
        BUILTIN_NAME,
        &ATAN2_INTEGER_INPUT_EXTENSION,
        &ATAN2_LOGICAL_INPUT_EXTENSION,
        &ATAN2_CHARACTER_INPUT_EXTENSION,
    )
}

fn value_into_atan2_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars
                .data()
                .iter()
                .map(|&ch| u32::from(ch.raw()) as f64)
                .collect();
            Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, e))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(atan2_error(&ATAN2_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::GpuTensor(_) => Err(atan2_error_with_detail(
            &ATAN2_ERROR_INTERNAL,
            "internal error converting GPU tensor",
        )),
        other => tensor::value_into_tensor_for("atan2", other)
            .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_INVALID_INPUT, e)),
    }
}

#[cfg(test)]
#[path = "atan2/tests.rs"]
pub(crate) mod tests;
