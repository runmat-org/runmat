//! MATLAB-compatible `ge` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::RelationalOperator;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::logical::rel::comparison;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::ge")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ge",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_ge",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers provider elem_ge kernels; complex-interleaved inputs compare provider-extracted real lanes, and unsupported routes gather to authoritative host storage.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::ge")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ge",
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
            Ok(format!("select({zero}, {one}, ({lhs} >= {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits comparison kernels that write 1 when the left operand is greater than or equal to the right.",
};

#[runtime_builtin(
    name = "ge",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::rel::ge"
)]
async fn ge_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    comparison::evaluate(lhs, rhs, RelationalOperator::GreaterThanOrEqual).await
}

#[cfg(test)]
mod tests;
