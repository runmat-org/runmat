use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::io::repl_fs::cd::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cd", op_kind: GpuOpKind::Custom("io"), supported_precisions: &[],
    broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately, nan_mode: ReductionNaN::Include,
    two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false,
    notes: "Working-folder queries and mutations accept host text and never invoke an accelerator provider.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::io::repl_fs::cd::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Environment mutations are fusion boundaries.",
};
