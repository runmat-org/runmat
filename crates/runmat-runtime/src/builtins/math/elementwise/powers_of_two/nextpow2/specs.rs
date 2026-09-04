use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::BUILTIN_NAME;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::powers_of_two::nextpow2"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_nextpow2",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate floating next-power-of-two exponents directly; unsupported operations use exact-owner host fallback and validated restoration.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::powers_of_two::nextpow2"
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
                "select(ceil(log2(abs({input}))), 0.0, abs({input}) == 0.0)"
            ))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion evaluates ceil(log2(abs(x))) and maps zero to zero.",
};
