use crate::builtins::common::spec::*;
#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::fullfile::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "fullfile", op_kind: GpuOpKind::Custom("io"), supported_precisions: &[], broadcast: BroadcastSemantics::None, provider_hooks: &[], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::GatherImmediately, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Lexical path assembly is host-owned; only the RunMat numeric character-code extension may gather an eligible resident row." };
#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::fullfile::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fullfile",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text-container path assembly is a fusion boundary.",
};
