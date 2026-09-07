//! Element-wise exponentiation binding and composition.

use runmat_builtins::{
    BuiltinErrorDescriptor, POWER_CATALOG_ENTRY, POWER_ERROR_INVALID_ARGUMENT,
    POWER_ERROR_INVALID_INPUT, POWER_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_arithmetic::reject_integer_logical_operands;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::power"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "power",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_pow",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers execute floating element-wise pow when both operands reside on the device; integer operands gather for exact exponent-domain validation and arithmetic.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::power"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "power",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let base = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let exp = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("pow({base}, {exp})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion planner lowers A.^B into WGSL pow() when both inputs are real; complex fallbacks execute on the host.",
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
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::power"
)]
async fn power_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    let template = parse_output_template(OUTPUT_PROTOTYPE_CONTEXT, &rest)?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &POWER_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let base_result = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => power_gpu_pair(la, lb).await,
        (Value::GpuTensor(la), rhs) => power_gpu_host_left(la, rhs).await,
        (lhs, Value::GpuTensor(rb)) => power_gpu_host_right(lhs, rb).await,
        (lhs, rhs) => Ok(power_host(lhs, rhs)?),
    }?;
    apply_output_template(OUTPUT_PROTOTYPE_CONTEXT, base_result, &template).await
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
use host::power_host;

mod math;
use math::{complex_pow_scalar, complex_pow_scalar_f32};

mod provider;
use provider::{power_gpu_host_left, power_gpu_host_right, power_gpu_pair};

#[cfg(test)]
pub(crate) mod tests;
