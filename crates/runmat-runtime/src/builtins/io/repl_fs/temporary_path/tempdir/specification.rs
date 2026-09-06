use crate::builtins::common::spec::*;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::io::repl_fs::temporary_path::tempdir::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tempdir",
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
    notes: "The session temporary-directory query is host-owned and has no provider inputs.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::io::repl_fs::temporary_path::tempdir::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tempdir",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Session environment reads are fusion boundaries.",
};
