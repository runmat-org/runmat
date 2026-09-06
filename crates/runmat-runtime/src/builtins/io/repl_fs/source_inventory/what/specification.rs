use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::io::repl_fs::source_inventory::what::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "what",
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
    notes: "Filesystem inventory executes on the host. Resident arguments reject before provider access.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::io::repl_fs::source_inventory::what::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "what",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filesystem inventory is an immediate read boundary and is not fusible.",
};
