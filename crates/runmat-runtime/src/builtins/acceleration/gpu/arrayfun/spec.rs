use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::arrayfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "arrayfun",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_sin" },
        ProviderHook::Unary { name: "unary_cos" },
        ProviderHook::Unary { name: "unary_abs" },
        ProviderHook::Unary { name: "unary_exp" },
        ProviderHook::Unary { name: "unary_log" },
        ProviderHook::Unary { name: "unary_sqrt" },
        ProviderHook::Binary {
            name: "elem_add",
            commutative: true,
        },
        ProviderHook::Binary {
            name: "elem_sub",
            commutative: false,
        },
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers that implement the listed kernels can run supported callbacks entirely on the GPU; unsupported callbacks fall back to the host path with re-upload.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::arrayfun"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "arrayfun",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because the callback can run arbitrary MATLAB code.",
};
