use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::exponentials::exp"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "exp",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_exp" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate real floating exp directly; complex values and unsupported hooks use an owner-preserving host fallback.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::exponentials::exp"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "exp",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |context: &FusionExprContext| {
            let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits the provider's elementwise exponential operation.",
};
