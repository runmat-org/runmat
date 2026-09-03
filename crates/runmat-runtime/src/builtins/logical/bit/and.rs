//! Element-wise logical `and` runtime binding.

use runmat_builtins::LogicalBinaryOperator;
#[cfg(test)]
use runmat_builtins::{AND_ERROR_INVALID_INPUT, AND_ERROR_SIZE_MISMATCH, AND_EXTENSIONS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::bit::and")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "and",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "logical_and",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to host execution when the provider does not implement logical_and; non-zero (including NaN) inputs map to true.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::bit::and")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "and",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let cond = format!("(({lhs} != {zero}) && ({rhs} != {zero}))");
            Ok(format!("select({zero}, {one}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion generates WGSL kernels that treat non-zero inputs as true and write 0/1 outputs.",
};

#[runtime_builtin(
    name = "and",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::bit::and"
)]
async fn and_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    super::truth::binary(lhs, rhs, LogicalBinaryOperator::And).await
}

#[cfg(all(test, feature = "wgpu"))]
async fn and_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    super::truth::binary(lhs, rhs, LogicalBinaryOperator::And).await
}

#[cfg(test)]
mod tests;
