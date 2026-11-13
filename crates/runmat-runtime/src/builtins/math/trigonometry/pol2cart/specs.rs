use super::*;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::pol2cart")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_cos" },
        ProviderHook::Unary { name: "unary_sin" },
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat keeps real gpuArray coordinate transforms resident by composing provider sin/cos and broadcasted multiply hooks. Providers without those hooks fall back to the host implementation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::pol2cart"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pol2cart returns multiple outputs and is treated as a terminal coordinate transform.",
};
