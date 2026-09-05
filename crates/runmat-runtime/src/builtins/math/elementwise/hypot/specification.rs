use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::hypot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hypot",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_hypot",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can execute hypot in a single binary kernel; the runtime gathers to host when the hook is unavailable or shapes require implicit expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::hypot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hypot",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |context: &FusionExprContext| {
            let left = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let right = context.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("hypot({left}, {right})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits WGSL hypot(a, b); providers may override via elem_hypot.",
};
