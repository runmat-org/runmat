//! Element-wise logical `or` runtime binding.

use runmat_builtins::LogicalBinaryOperator;
#[cfg(test)]
use runmat_builtins::{OR_ERROR_INVALID_INPUT, OR_ERROR_SIZE_MISMATCH};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::bit::or")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "or",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "logical_or",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses logical_or only for ownership-validated floating real handles. Native integer handles gather through exact typed storage; explicit gpuArray fallback restores a validated logical result while automatic residency may remain on host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::bit::or")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "or",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let one = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let cond = format!("(({lhs} != {zero}) || ({rhs} != {zero}))");
            Ok(format!("select({zero}, {one}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion generates WGSL kernels that treat non-zero inputs as true and write 0/1 outputs.",
};

#[runtime_builtin(
    name = "or",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::bit::or"
)]
async fn or_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    super::truth::binary(lhs, rhs, LogicalBinaryOperator::Or).await
}

#[cfg(all(test, feature = "wgpu"))]
async fn or_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    super::truth::binary(lhs, rhs, LogicalBinaryOperator::Or).await
}

#[cfg(test)]
mod tests;
