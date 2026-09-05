use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::BUILTIN_NAME;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::heaviside::specification"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_heaviside",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may evaluate the real step operation on device; typed unsupported hooks use an exact-owner host fallback.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::heaviside::specification"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: expression,
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion emits the same H(0) = 0.5 and NaN-preserving mapping as host execution.",
};

fn expression(context: &FusionExprContext) -> Result<String, FusionError> {
    let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
    let (zero, half, one) = match context.scalar_ty {
        ScalarType::F32 => ("0.0", "0.5", "1.0"),
        ScalarType::F64 => ("f64(0.0)", "f64(0.5)", "f64(1.0)"),
        ScalarType::I32 | ScalarType::Bool => {
            return Err(FusionError::UnsupportedPrecision(context.scalar_ty));
        }
    };
    Ok(format!(
        "select(select(select({zero}, {one}, ({input} > {zero})), {half}, ({input} == {zero})), {input}, isNan({input}))"
    ))
}
