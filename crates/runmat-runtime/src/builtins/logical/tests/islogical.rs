//! Logical-storage metadata predicate.

use super::metadata::MetadataBoundary;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISLOGICAL_CATALOG_ENTRY, ISLOGICAL_ERROR_INTERNAL,
    ISLOGICAL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "islogical", op_kind: GpuOpKind::Custom("metadata"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::GatherImmediately, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Validates exact-owner class metadata and returns a host logical scalar without gathering payload data." };
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "islogical",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata query that forms a fusion boundary.",
};
const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISLOGICAL_CATALOG_ENTRY,
    &ISLOGICAL_ERROR_INTERNAL,
    &ISLOGICAL_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::Logical,
);
#[runtime_builtin(
    name = "islogical",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::islogical"
)]
async fn islogical_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}
#[cfg(test)]
#[path = "islogical/tests.rs"]
mod tests;
