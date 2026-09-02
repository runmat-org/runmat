//! Element-wise finite-value classification.

use runmat_builtins::{
    NumericClassificationPredicate, ISFINITE_CATALOG_ENTRY, ISFINITE_ERROR_INTERNAL,
    ISFINITE_ERROR_INVALID_INPUT, ISFINITE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::classification::ClassificationBoundary;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isfinite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isfinite",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isfinite",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses the exact provider's logical-isfinite operation and enters class-preserving fallback only for a typed unsupported response.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isfinite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isfinite",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |context: &FusionExprContext| {
            let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match context.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("select({zero}, {one}, isFinite({input}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit zero/one logical masks.",
};

const BOUNDARY: ClassificationBoundary = ClassificationBoundary::new(
    &ISFINITE_CATALOG_ENTRY,
    &ISFINITE_ERROR_INVALID_INPUT,
    &ISFINITE_ERROR_INTERNAL,
    &ISFINITE_ERROR_TOO_MANY_OUTPUTS,
    NumericClassificationPredicate::Finite,
);

#[runtime_builtin(
    name = "isfinite",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::isfinite"
)]
async fn isfinite_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value).await
}

#[cfg(test)]
#[path = "isfinite/tests.rs"]
mod tests;
