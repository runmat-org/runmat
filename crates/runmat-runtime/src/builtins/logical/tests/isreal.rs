//! Real-storage metadata predicate.

use super::metadata::MetadataBoundary;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISREAL_CATALOG_ENTRY, ISREAL_ERROR_INTERNAL, ISREAL_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isreal")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isreal",
    op_kind: GpuOpKind::Custom("storage-check"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Validates exact-owner storage and class metadata without reading resident payload data.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isreal")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isreal",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata query that forms a fusion boundary.",
};
const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISREAL_CATALOG_ENTRY,
    &ISREAL_ERROR_INTERNAL,
    &ISREAL_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::Real,
);
#[runtime_builtin(
    name = "isreal",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::isreal"
)]
async fn isreal_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}
#[cfg(test)]
#[path = "isreal/tests.rs"]
mod tests;
