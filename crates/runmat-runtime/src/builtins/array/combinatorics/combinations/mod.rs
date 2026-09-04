//! `combinations` runtime identity and composition root.

mod arguments;
mod columns;
mod error;
mod execute;
mod plan;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;

const NAME: &str = "combinations";

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::combinatorics::combinations"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("table_construct"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "combinations gathers compatibility-gated resident inputs and constructs its table on the host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::combinatorics::combinations"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "combinations materializes a Cartesian-product table and is not fusion eligible.",
};

#[runtime_builtin(
    name = "combinations",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::combinatorics::combinations"
)]
async fn combinations_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    execute::apply(first, rest).await
}
