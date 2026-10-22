//! Sampled-data trapezoidal integration through shared borrowed admission.
use super::integration_common::{
    self, canonical_shape_complex, canonical_shape_tensor, default_dimension_from_shape,
    dim_product, interval_width, pad_shape_for_dim, spacing_from_value, value_into_complex_tensor,
    IntegrationArguments, SpacingSpec,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, TRAPZ_DESCRIPTOR, TRAPZ_ERROR_INTERNAL, TRAPZ_ERROR_INVALID_ARGUMENT,
    TRAPZ_ERROR_INVALID_INPUT,
};
#[cfg(test)]
use runmat_builtins::{TRAPZ_INTEGER_DATA_EXTENSION, TRAPZ_INTEGER_SPACING_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};
const NAME: &str = "trapz";
mod admission;
mod host;
use host::trapz_complex_tensor;
pub(crate) use host::trapz_tensor;
#[cfg(test)]
pub(crate) mod tests;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::trapz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "trapz",
    op_kind: GpuOpKind::Custom("trapezoidal-integral"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("trapz_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Eligible floating real or complex samples use the exact owning provider's trapz_dim hook with supported real spacing. Unsupported automatic placement may fall back before an attempt; explicit input is not silently gathered and attempted failures are terminal.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::trapz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "trapz",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Discrete integration currently lowers to the runtime implementation rather than fusion kernels.",
};

fn trapz_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    trapz_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}
fn trapz_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn trapz_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    trapz_error_with_detail(&TRAPZ_ERROR_INTERNAL, detail)
}
#[runtime_builtin(
    name = "trapz",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::trapz"
)]
async fn trapz_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &TRAPZ_DESCRIPTOR,
        &runmat_builtins::TRAPZ_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )
    .map_err(super::numerical_execution::terminal)?;
    let call = super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::Trapezoid,
        runtime.as_ref(),
    )?;

    let args = integration_common::parse_integration_arguments(call.kind, &first, rest.view())?;
    admission::admit(&args)?;
    integration_common::validate_integration_inputs(call.kind, &args)?;
    if let Some(output) = integration_common::try_provider_integration(&args, &call).await? {
        return call.finish_value(Ok(output));
    }
    let (y, spacing) = integration_common::gather_integration_inputs(&args, &call).await?;
    let result = match y {
        Value::Complex(..) | Value::ComplexTensor(_) => {
            let input = value_into_complex_tensor(NAME, y)?;
            let shape = canonical_shape_complex(&input);
            let dimension = args
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing = spacing_from_value(NAME, spacing, &shape, dimension)
                .map_err(|e| trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, e.message()))?;
            complex_tensor_into_value(trapz_complex_tensor(&input, &spacing, dimension)?)
        }
        other => {
            let input = tensor::value_into_tensor_for(NAME, other)
                .map_err(|e| trapz_error_with_detail(&TRAPZ_ERROR_INVALID_INPUT, e))?;
            let shape = canonical_shape_tensor(&input);
            let dimension = args
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing = spacing_from_value(NAME, spacing, &shape, dimension)
                .map_err(|e| trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, e.message()))?;
            tensor::tensor_into_value(trapz_tensor(&input, &spacing, dimension)?)
        }
    };
    integration_common::restore_integration_result(result, &args, &call)
}
