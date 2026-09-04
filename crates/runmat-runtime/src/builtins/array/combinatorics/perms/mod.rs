//! `perms` runtime identity and composition root.

mod cardinality;
mod containers;
mod error;
mod execute;
mod shape;

#[cfg(test)]
mod tests;

use runmat_builtins::PERMS_ERROR_INVALID_INPUT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;

const NAME: &str = "perms";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::combinatorics::perms")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Deterministic permutation materialization runs on the host and restores provider-resident outputs through their input owner.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::combinatorics::perms"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "perms materializes factorial-size output and is not fusion eligible.",
};

#[runtime_builtin(
    name = "perms",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::combinatorics::perms"
)]
async fn perms_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(error::with_message(
            &PERMS_ERROR_INVALID_INPUT,
            "perms: too many input arguments",
        ));
    }
    execute::apply(value).await
}
