//! Element-wise left-division binding and composition.

use runmat_builtins::{
    LDIVIDE_CATALOG_ENTRY, LDIVIDE_ERROR_INTERNAL, LDIVIDE_ERROR_INVALID_ARGUMENT,
    LDIVIDE_ERROR_INVALID_INPUT, LDIVIDE_ERROR_SIZE_MISMATCH, LDIVIDE_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_arithmetic::reject_integer_logical_operands;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::ldivide"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ldivide",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
        ProviderHook::Custom("scalar_div"),
        ProviderHook::Custom("scalar_rdiv"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses validated exact-owner elem_div for supported same-class resident pairs, including provider-supported implicit expansion and typed-integer division; scalar_div/scalar_rdiv retain floating scalar cases, while exact integer scalar values are uploaded in the resident integer class. Unsupported or mixed-class cases use a class-preserving host fallback before 'like' prototypes are honoured.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::ldivide"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ldivide",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let divisor = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let numerator = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({numerator} / {divisor})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain quotient; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "ldivide";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ldivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::ldivide"
)]
async fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    let template = parse_output_template(OUTPUT_PROTOTYPE_CONTEXT, &rest)?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LDIVIDE_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let base = super::division::execute(DIVISION_CONTEXT, rhs, lhs).await?;
    apply_output_template(OUTPUT_PROTOTYPE_CONTEXT, base, &template).await
}

#[cfg(test)]
use super::output_prototype::real_to_complex;
use super::output_prototype::{
    apply_output_template, parse_output_template, OutputPrototypeContext, OutputTemplate,
};

const OUTPUT_PROTOTYPE_CONTEXT: OutputPrototypeContext = OutputPrototypeContext {
    identity: LDIVIDE_CATALOG_ENTRY.identity,
    invalid_argument: &LDIVIDE_ERROR_INVALID_ARGUMENT,
    invalid_input: &LDIVIDE_ERROR_INVALID_INPUT,
};

use super::division::DivisionContext;

const DIVISION_CONTEXT: DivisionContext = DivisionContext {
    identity: LDIVIDE_CATALOG_ENTRY.identity,
    invalid_input: &LDIVIDE_ERROR_INVALID_INPUT,
    size_mismatch: &LDIVIDE_ERROR_SIZE_MISMATCH,
    internal: &LDIVIDE_ERROR_INTERNAL,
};

#[cfg(test)]
pub(crate) mod tests;
