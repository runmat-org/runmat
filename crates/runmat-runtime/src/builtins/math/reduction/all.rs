//! Logical AND reduction builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    LogicalReductionKind, ALL_CATALOG_ENTRY, ALL_ERROR_INTERNAL, ALL_ERROR_INVALID_ARGUMENT,
    ALL_ERROR_INVALID_INPUT, ALL_ERROR_TOO_MANY_OUTPUTS, ALL_NANFLAG_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const CONFIG: super::logical::LogicalReductionConfig = super::logical::LogicalReductionConfig {
    entry: &ALL_CATALOG_ENTRY,
    kind: LogicalReductionKind::All,
    invalid_argument: &ALL_ERROR_INVALID_ARGUMENT,
    invalid_input: &ALL_ERROR_INVALID_INPUT,
    internal: &ALL_ERROR_INTERNAL,
    too_many_outputs: &ALL_ERROR_TOO_MANY_OUTPUTS,
    nanflag_extension: &ALL_NANFLAG_EXTENSION,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::all")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "all",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_all_dim",
        },
        ProviderHook::Reduction { name: "reduce_all" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may execute AND reductions; unsupported hooks use the exact host path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::all")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "all",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |context: &FusionExprContext| {
            let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "accumulator *= select(0.0, 1.0, ({input} != 0.0) || ({input} != {input}));"
            ))
        },
    }),
    emits_nan: false,
    notes: "Fusion reductions treat NaN as nonzero and combine truth values with AND.",
};

#[runtime_builtin(
    name = "all",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::all"
)]
async fn all_builtin(value: Value, arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::logical::execute(&CONFIG, value, arguments).await
}

#[cfg(all(test, feature = "wgpu"))]
use super::logical::arguments::ReductionSpec;
#[cfg(test)]
use runmat_builtins::ALL_DESCRIPTOR;
#[cfg(all(test, feature = "wgpu"))]
async fn all_host(
    value: Value,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    super::logical::host::reduce(&CONFIG, value, spec, nan_mode).await
}

#[cfg(test)]
#[path = "all/tests.rs"]
pub(crate) mod tests;
