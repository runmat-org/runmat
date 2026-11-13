//! Element-wise right-division binding and composition.

use runmat_builtins::{
    RDIVIDE_CATALOG_ENTRY, RDIVIDE_ERROR_INTERNAL, RDIVIDE_ERROR_INVALID_ARGUMENT,
    RDIVIDE_ERROR_INVALID_INPUT, RDIVIDE_ERROR_SIZE_MISMATCH, RDIVIDE_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::elementwise::integer_arithmetic::reject_integer_logical_operands;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::rdivide"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rdivide",
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
    notes: "Uses elem_div when shapes match, including complex-interleaved gpuArrays, attempts provider-side implicit expansion with repmat, and uses scalar_div/scalar_rdiv for real scalar cases; unsupported shapes or operand kinds fall back to the CPU before 'like' prototypes are honoured.",
};

const FUSION_SEMANTICS: crate::builtins::common::spec::FusionSemanticDeclaration =
    crate::builtins::common::spec::FusionSemanticDeclaration {
        binding: runmat_builtins::BuiltinBindingIdentity {
            builtin: RDIVIDE_CATALOG_ENTRY.identity,
            variant: "default",
        },
        revision: 1,
        admit: None,
        admit_static: None,
        helpers: &[],
        required_real_unary: None,
        required_arithmetic: None,
        scalar_operation: Some(runmat_types::ScalarOperation::Operator(
            runmat_types::OperatorKind::ElementwiseDivide,
        )),
    };

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::rdivide",
    semantics = FUSION_SEMANTICS,
    host_resources = crate::builtins::math::host_resources::real_binary
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rdivide",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_real_binary_expression(
                runmat_types::OperatorKind::ElementwiseDivide, ctx, writer,
            )
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain quotient; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "rdivide";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "rdivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::rdivide"
)]
async fn rdivide_builtin(
    runtime: Option<RuntimeContext>,
    lhs: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&lhs, &rhs].into_iter().chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::RDIVIDE_DESCRIPTOR,
        &runmat_builtins::RDIVIDE_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    let template = parse_output_template(OUTPUT_PROTOTYPE_CONTEXT, rest.view())?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RDIVIDE_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let template =
        super::output_prototype::resolve_output_template(OUTPUT_PROTOTYPE_CONTEXT, &template)
            .await?;
    let provider_runtime = runtime.clone();
    super::preparation::execute(
        runtime.as_ref(),
        RDIVIDE_CATALOG_ENTRY.identity,
        runmat_types::OperatorKind::ElementwiseDivide,
        2 + rest.len(),
        lhs,
        rhs,
        |lhs, rhs| async move {
            let base =
                super::division::execute(provider_runtime.as_ref(), DIVISION_CONTEXT, lhs, rhs)
                    .await?;
            let base = RootedValue::new(base).map_err(|cause| cause.into_runtime_error())?;
            let value =
                apply_output_template(OUTPUT_PROTOTYPE_CONTEXT, base.value().clone(), &template)
                    .await?;
            RootedValue::new(value).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}

#[cfg(test)]
use super::output_prototype::real_to_complex;
use super::output_prototype::{
    apply_output_template, parse_output_template, OutputPrototypeContext, OutputTemplate,
};

const OUTPUT_PROTOTYPE_CONTEXT: OutputPrototypeContext = OutputPrototypeContext {
    identity: RDIVIDE_CATALOG_ENTRY.identity,
    invalid_argument: &RDIVIDE_ERROR_INVALID_ARGUMENT,
    invalid_input: &RDIVIDE_ERROR_INVALID_INPUT,
};

use super::division::{DivisionContext, PublicOperandOrder};

const DIVISION_CONTEXT: DivisionContext = DivisionContext {
    entry: &RDIVIDE_CATALOG_ENTRY,
    public_order: PublicOperandOrder::Numerical,
    identity: RDIVIDE_CATALOG_ENTRY.identity,
    invalid_input: &RDIVIDE_ERROR_INVALID_INPUT,
    size_mismatch: &RDIVIDE_ERROR_SIZE_MISMATCH,
    internal: &RDIVIDE_ERROR_INTERNAL,
};

#[cfg(test)]
pub(crate) mod tests;
