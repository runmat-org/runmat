use crate::builtins::common::spec::*;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::io::repl_fs::directory_listing::ls::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ls",
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
    notes:
        "Host-text filesystem input only; provider-resident values reject before provider access.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::io::repl_fs::directory_listing::ls::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ls",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filesystem inspection is a fusion boundary.",
};
