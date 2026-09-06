use crate::builtins::common::spec::*;
#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::filesep::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filesep",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The platform file separator is a host constant with no provider inputs.",
};
#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::filesep::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filesep",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "The platform file separator is emitted outside fused numerical regions.",
};
