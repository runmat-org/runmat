use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::complex_construction::complex"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "complex",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::ScalarOnly,
    provider_hooks: &[
        ProviderHook::Unary { name: "complex_from_real" },
        ProviderHook::Binary { name: "complex_from_real_imag", commutative: false },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may construct complex-interleaved floating storage. Exact fixed-width integer construction uses owner-aware host fallback and class-preserving restoration.",
};
