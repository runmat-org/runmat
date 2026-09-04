use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::BUILTIN_NAME;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::powers_of_two::pow2"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_pow2" },
        ProviderHook::Binary {
            name: "pow2_scale",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate unary powers and shape-matched binary scaling; other forms use the canonical host implementation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::powers_of_two::pow2"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |context: &FusionExprContext| {
            let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "exp({input} * {:.17})",
                std::f64::consts::LN_2
            ))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Unary floating pow2 can fuse as exp(x * ln(2)); binary scaling remains a provider or runtime operation.",
};
