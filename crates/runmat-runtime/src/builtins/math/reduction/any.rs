//! Logical OR reduction builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    LogicalReductionKind, ANY_CATALOG_ENTRY, ANY_ERROR_INTERNAL, ANY_ERROR_INVALID_ARGUMENT,
    ANY_ERROR_INVALID_INPUT, ANY_ERROR_TOO_MANY_OUTPUTS, ANY_NANFLAG_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const CONFIG: super::logical::LogicalReductionConfig = super::logical::LogicalReductionConfig {
    entry: &ANY_CATALOG_ENTRY,
    kind: LogicalReductionKind::Any,
    invalid_argument: &ANY_ERROR_INVALID_ARGUMENT,
    invalid_input: &ANY_ERROR_INVALID_INPUT,
    internal: &ANY_ERROR_INTERNAL,
    too_many_outputs: &ANY_ERROR_TOO_MANY_OUTPUTS,
    nanflag_extension: &ANY_NANFLAG_EXTENSION,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::any")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "any",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_any_dim",
        },
        ProviderHook::Reduction { name: "reduce_any" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Omit,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may execute OR reductions; unsupported hooks and omit-NaN forms use the exact host path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::any")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "any",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |context: &FusionExprContext| {
            let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "accumulator = max(accumulator, select(0.0, 1.0, ({input} != 0.0) && ({input} == {input})));"
            ))
        },
    }),
    emits_nan: false,
    notes: "Fusion reductions apply the default omit-NaN policy and combine truth values with OR.",
};

#[runtime_builtin(
    name = "any",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::any"
)]
async fn any_builtin(value: Value, arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::logical::execute(&CONFIG, value, arguments).await
}

#[cfg(all(test, feature = "wgpu"))]
use super::logical::arguments::ReductionSpec;
#[cfg(test)]
use runmat_builtins::ANY_DESCRIPTOR;
#[cfg(all(test, feature = "wgpu"))]
async fn any_host(
    value: Value,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    super::logical::host::reduce(&CONFIG, value, spec, nan_mode).await
}

#[cfg(test)]
#[path = "any/tests.rs"]
pub(crate) mod tests;
