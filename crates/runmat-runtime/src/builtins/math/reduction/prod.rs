//! Product reduction with native arithmetic and an exact owning-provider boundary.
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, ProductOutputClass as OutputTemplate, PROD_DESCRIPTOR,
    PROD_ERROR_COMPLEX_UNSUPPORTED, PROD_ERROR_INTERNAL, PROD_ERROR_INVALID_ARGUMENT,
    PROD_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};
const NAME: &str = "prod";
mod arguments;
mod geometry;
mod host;
mod provider;
use arguments::{DimSelection, ParsedArguments};
use geometry::ResolvedDims;
#[cfg(test)]
pub(crate) mod tests;
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::prod")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "prod",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_prod_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_prod",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes:
        "Eligible floating products and native integer products use distinct owning-provider hooks. Automatic placement may fall back before the first provider attempt; execution errors and later pipeline declines are terminal. Unsupported explicit input is not gathered.",
};

fn prod_descriptor_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    prod_descriptor_error_with_message(error.message, error)
}
fn prod_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    prod_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}
fn prod_descriptor_error_with_message(
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
fn prod_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    prod_descriptor_error_with_detail(&PROD_ERROR_INTERNAL, detail)
}
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::prod")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "prod",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            writer.write_formatted(format_args!("accumulator *= {input};"))
        }),
    }),
    emits_nan: false,
    notes: "Product shader templates are retained, but no typed product reduction is admitted by the selected fusion binding. Calls execute through the Runtime boundary.",
};

#[runtime_builtin(
    name = "prod",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::prod"
)]
async fn prod_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &PROD_DESCRIPTOR,
        &runmat_builtins::PROD_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )
    .map_err(super::numerical_execution::terminal)?;
    let call = super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::Product,
        runtime.as_ref(),
    )?;
    if matches!(value, Value::Complex(..) | Value::ComplexTensor(_))
        || matches!(&value,Value::GpuTensor(h) if runmat_accelerate_api::handle_storage(h)==runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
    {
        return Err(prod_descriptor_error(&PROD_ERROR_COMPLEX_UNSUPPORTED));
    }
    if !matches!(
        value,
        Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::GpuTensor(_)
    ) {
        return Err(prod_descriptor_error(&PROD_ERROR_INVALID_INPUT));
    }
    let parsed = arguments::parse(rest.view(), &call).await?;
    if let Some(result) = host::scalar(&value, &parsed) {
        return call.finish_value(Ok(result));
    }
    if let Value::GpuTensor(handle) = &value {
        gpu_helpers::require_resident_compute_owner(handle, || {
            prod_internal_error("resident product input has no exact owning provider")
        })?;
    }
    let result = match &value {
        Value::GpuTensor(handle) => provider::execute(handle, &parsed, &call).await,
        _ => {
            call.host()?;
            host::execute(value, &parsed)
        }
    };
    call.finish_value(result)
}

pub(in crate::builtins::math) async fn product_geometry(
    shape: &[usize],
    arguments: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<super::sum::ReductionGeometry> {
    let call = super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::Product,
        runtime,
    )?;
    call.host()?;
    let parsed = arguments::parse(arguments, &call).await?;
    call.check()?;
    let axes = ResolvedDims::resolve(shape, &parsed.selection);
    let axes = axes.iter().collect::<Vec<_>>();
    super::sum::reduction_geometry_for_axes(shape, &axes)
}
