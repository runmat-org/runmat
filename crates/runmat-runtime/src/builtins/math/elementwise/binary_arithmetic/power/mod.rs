//! Element-wise exponentiation binding and composition.

use runmat_builtins::{
    BuiltinErrorDescriptor, POWER_CATALOG_ENTRY, POWER_ERROR_INVALID_ARGUMENT,
    POWER_ERROR_INVALID_INPUT, POWER_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_arithmetic::reject_integer_logical_operands;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::power"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "power",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Custom("classify_real_power"),
        ProviderHook::Custom("real_power_in_domain"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real floating pairs are classified at their actual broadcast positions before domain-preserving provider power; integer operands retain the exact host exponent-domain and arithmetic path.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::power",
    semantics = fusion::SEMANTICS,
    epilogue = crate::builtins::common::spec::FusionEpilogueOperation::RealPower,
    host_resources = crate::builtins::math::host_resources::real_power
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "power",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_real_power_expression(ctx, writer)
        }),
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion requires exact paired real-domain admission and the shared exceptional-value-correct power helper. Unknown intermediate pairs and complex results retain shared runtime execution.",
};

const BUILTIN_NAME: &str = "power";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("power").build()
}

fn power_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn power_error_with_detail(
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

#[runtime_builtin(
    name = "power",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::power"
)]
async fn power_builtin(
    runtime: Option<RuntimeContext>,
    lhs: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&lhs, &rhs].into_iter().chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::POWER_DESCRIPTOR,
        &runmat_builtins::POWER_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    let template = parse_output_template(OUTPUT_PROTOTYPE_CONTEXT, rest.view())?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &POWER_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let template =
        super::output_prototype::resolve_output_template(OUTPUT_PROTOTYPE_CONTEXT, &template)
            .await?;
    let provider_runtime = runtime.clone();
    preparation::execute(
        runtime.as_ref(),
        2 + rest.len(),
        lhs,
        rhs,
        |lhs, rhs| async move {
            let base_result = match (lhs, rhs) {
                (Value::GpuTensor(la), Value::GpuTensor(lb)) => {
                    power_gpu_pair(provider_runtime.as_ref(), la, lb).await
                }
                (Value::GpuTensor(la), rhs) => {
                    power_gpu_host_left(provider_runtime.as_ref(), la, rhs).await
                }
                (lhs, Value::GpuTensor(rb)) => {
                    power_gpu_host_right(provider_runtime.as_ref(), lhs, rb).await
                }
                (lhs, rhs) => Ok(power_host(lhs, rhs)?),
            }?;
            RootedValue::new(base_result).map_err(|cause| cause.into_runtime_error())
        },
        |base| async move {
            if matches!(
                &template,
                super::output_prototype::ResolvedOutputTemplate::Default
            ) {
                return Ok(base);
            }
            let value =
                apply_output_template(OUTPUT_PROTOTYPE_CONTEXT, base.value().clone(), &template)
                    .await?;
            RootedValue::new(value).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}

#[cfg(test)]
use super::output_prototype::real_to_complex;
use super::output_prototype::{
    apply_output_template, parse_output_template, OutputPrototypeContext, OutputTemplate,
};

const OUTPUT_PROTOTYPE_CONTEXT: OutputPrototypeContext = OutputPrototypeContext {
    identity: POWER_CATALOG_ENTRY.identity,
    invalid_argument: &POWER_ERROR_INVALID_ARGUMENT,
    invalid_input: &POWER_ERROR_INVALID_INPUT,
};

mod host;
mod domain;
mod fusion;
mod preparation;
use host::power_host;

mod math;
use math::{complex_pow_scalar, complex_pow_scalar_f32};

mod provider;
mod compound;
mod scalar_compound;
use provider::{power_gpu_host_left, power_gpu_host_right, power_gpu_pair};

#[cfg(test)]
pub(crate) mod tests;
