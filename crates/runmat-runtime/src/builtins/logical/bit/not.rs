//! Element-wise logical `not` runtime binding.

use runmat_builtins::LogicalUnaryOperator;
#[cfg(test)]
use runmat_builtins::NOT_ERROR_INVALID_INPUT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::bit::not")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "not",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_not",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses logical_not only for an ownership-validated floating real handle. Native integer handles gather through exact typed storage; explicit gpuArray fallback restores a validated logical result while automatic residency may remain on host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::bit::not")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "not",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0".to_string(), "1.0".to_string()),
                ScalarType::F64 => ("f64(0.0)".to_string(), "f64(1.0)".to_string()),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let cond = format!("({input} != {zero})");
            Ok(format!("select({one}, {zero}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat any non-zero input as true and write 0/1 outputs, matching MATLAB logical semantics.",
};

#[runtime_builtin(
    name = "not",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::bit::not"
)]
async fn not_builtin(value: Value) -> crate::BuiltinResult<Value> {
    super::truth::unary(value, LogicalUnaryOperator::Not).await
}

#[cfg(all(test, feature = "wgpu"))]
async fn not_host(value: Value) -> crate::BuiltinResult<Value> {
    super::truth::unary(value, LogicalUnaryOperator::Not).await
}

#[cfg(test)]
mod tests;
