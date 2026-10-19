//! Sparse-storage metadata predicate.

use super::metadata::MetadataBoundary;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISSPARSE_CATALOG_ENTRY, ISSPARSE_ERROR_INTERNAL,
    ISSPARSE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::issparse")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "issparse", op_kind: GpuOpKind::Custom("metadata"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::GatherImmediately, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Validates resident dense storage metadata without gathering; current gpuArray storage is not sparse." };
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::issparse")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issparse",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata query that forms a fusion boundary.",
};
const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISSPARSE_CATALOG_ENTRY,
    &ISSPARSE_ERROR_INTERNAL,
    &ISSPARSE_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::Sparse,
);
#[runtime_builtin(
    name = "issparse",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::issparse"
)]
async fn issparse_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}
#[cfg(test)]
#[path = "issparse/tests.rs"]
mod tests;
