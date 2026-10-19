//! Explicit gpuArray identity predicate.

use super::metadata::MetadataBoundary;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    MetadataPredicate, ISGPUARRAY_CATALOG_ENTRY, ISGPUARRAY_ERROR_INTERNAL,
    ISGPUARRAY_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isgpuarray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isgpuarray",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reports explicit gpuArray identity without gathering or inspecting payload data.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isgpuarray")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isgpuarray",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata query that forms a fusion boundary.",
};
const BOUNDARY: MetadataBoundary = MetadataBoundary::new(
    &ISGPUARRAY_CATALOG_ENTRY,
    &ISGPUARRAY_ERROR_INTERNAL,
    &ISGPUARRAY_ERROR_TOO_MANY_OUTPUTS,
    MetadataPredicate::GpuArray,
);
#[runtime_builtin(
    name = "isgpuarray",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::isgpuarray"
)]
async fn isgpuarray_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value)
}
#[cfg(test)]
#[path = "isgpuarray/tests.rs"]
mod tests;
