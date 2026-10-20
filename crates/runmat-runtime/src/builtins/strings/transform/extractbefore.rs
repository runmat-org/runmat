use super::compat::{boundary_transform, locate_boundary};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;
use runmat_builtins::{
    EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION, EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION,
    EXTRACT_BEFORE_STRING_CELL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::strings::transform::extractbefore"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "extractBefore",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host text operation. Resident numeric position controls gather only after RunMat extension admission; source text remains host-only.",
};
#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::extractbefore"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "extractBefore",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text transform executes on host and does not join fusion plans.",
};

#[runtime_builtin(
    name = "extractBefore",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::extractbefore"
)]
pub(super) async fn extract_before_builtin(
    text: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    boundary_transform(
        text,
        rest,
        "extractBefore",
        &EXTRACT_BEFORE_RESIDENT_POSITION_EXTENSION,
        &EXTRACT_BEFORE_CHAR_MATRIX_EXTENSION,
        &EXTRACT_BEFORE_STRING_CELL_EXTENSION,
        |s, boundary| {
            let (start, _) = locate_boundary(s, boundary)?;
            Ok(s[..start].to_string())
        },
    )
    .await
}
