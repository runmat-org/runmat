//! Numeric-storage metadata predicate.

use super::metadata::MetadataBoundary;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISNUMERIC_CATALOG_ENTRY, ISNUMERIC_ERROR_INTERNAL,
    ISNUMERIC_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "isnumeric", op_kind: GpuOpKind::Custom("metadata"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::GatherImmediately, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Validates exact-owner class metadata and returns a host logical scalar without gathering payload data." };
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isnumeric",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata query that forms a fusion boundary.",
};
const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISNUMERIC_CATALOG_ENTRY,
    &ISNUMERIC_ERROR_INTERNAL,
    &ISNUMERIC_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::Numeric,
);
#[runtime_builtin(
    name = "isnumeric",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::isnumeric"
)]
async fn isnumeric_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}
#[cfg(test)]
#[path = "isnumeric/tests.rs"]
mod tests;
