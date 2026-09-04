//! `nchoosek` runtime identity and composition root.

mod arguments;
mod coefficient;
mod combinations;
mod error;
mod execute;
mod numeric;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;

const NAME: &str = "nchoosek";

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::combinatorics::nchoosek"
)]
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
    notes: "nchoosek is a host combinatorial operation and rejects provider-resident inputs.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::combinatorics::nchoosek"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "nchoosek materializes combinatorial-size output and is not fusion eligible.",
};

#[runtime_builtin(
    name = "nchoosek",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::combinatorics::nchoosek"
)]
async fn nchoosek_builtin(first: Value, k: Value) -> BuiltinResult<Value> {
    execute::apply(first, k).await
}
