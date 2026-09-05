use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::logarithms::log"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec { name: "log", op_kind: GpuOpKind::Elementwise, supported_precisions: &[ScalarType::F32, ScalarType::F64], broadcast: BroadcastSemantics::Matlab, provider_hooks: &[ProviderHook::Unary { name: "unary_log" }], constant_strategy: ConstantStrategy::InlineLiteral, residency: ResidencyPolicy::NewHandle, nan_mode: ReductionNaN::Include, two_pass_threshold: None, workgroup_size: None, accepts_nan_mode: false, notes: "Nonnegative real inputs may remain resident; complex promotion uses an owner-preserving fallback." };
#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::logarithms::log"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real inputs can require complex promotion.",
};
