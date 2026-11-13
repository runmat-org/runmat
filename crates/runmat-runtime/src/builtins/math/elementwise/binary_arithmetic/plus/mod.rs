//! MATLAB-compatible `plus` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::PLUS_DESCRIPTOR;
use runmat_builtins::{
    PLUS_CATALOG_ENTRY, PLUS_ERROR_INTERNAL, PLUS_ERROR_INVALID_ARGUMENT, PLUS_ERROR_INVALID_INPUT,
    PLUS_ERROR_SIZE_MISMATCH, PLUS_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, NumericDType, NumericStorage, Tensor, Value,
};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{
    reject_integer_logical_operands, try_integer_binary, IntegerBinaryOp,
};
use crate::builtins::math::elementwise::sparse::{try_sparse_binary, SparseBinaryOp};
use crate::builtins::math::elementwise::sparse_integer::try_typed_sparse_integer_binary;
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;

mod error;
use error::{builtin_error, plus_error, plus_error_with_detail};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::plus"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "plus",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_add",
            commutative: true,
        },
        ProviderHook::Custom("scalar_add"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses elem_add for admitted real and complex-interleaved pairs; complex singleton expansion uses the provider broadcast hook directly, while real expansion can use repmat. Real scalar pairs can use scalar_add; unsupported hooks or operand forms fall back to host execution.",
};

const FUSION_SEMANTICS: crate::builtins::common::spec::FusionSemanticDeclaration =
    crate::builtins::common::spec::FusionSemanticDeclaration {
        binding: runmat_builtins::BuiltinBindingIdentity {
            builtin: PLUS_CATALOG_ENTRY.identity,
            variant: "default",
        },
        revision: 1,
        admit: None,
        admit_static: None,
        helpers: &[],
        required_real_unary: None,
        required_arithmetic: None,
        scalar_operation: Some(runmat_types::ScalarOperation::Operator(
            runmat_types::OperatorKind::Add,
        )),
    };

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::plus",
    semantics = FUSION_SEMANTICS,
    host_resources = crate::builtins::math::host_resources::real_binary
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "plus",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_real_binary_expression(
                runmat_types::OperatorKind::Add,
                ctx,
                writer,
            )
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion emits a plain sum; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "plus";

#[runtime_builtin(
    name = "plus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::plus"
)]
async fn plus_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&lhs, &rhs].into_iter().chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PLUS_DESCRIPTOR,
        &runmat_builtins::PLUS_ERROR_TOO_MANY_OUTPUTS,
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
    if matches!(lhs, Value::String(_) | Value::StringArray(_))
        || matches!(rhs, Value::String(_) | Value::StringArray(_))
    {
        if !rest.is_empty() {
            return Err(plus_error_with_detail(
                &PLUS_ERROR_INVALID_ARGUMENT,
                "numeric output prototypes do not apply to string addition",
            ));
        }
        let result = text::concatenate(runtime.as_ref(), &lhs, &rhs).await?;
        return RootedValue::new(result).map_err(|cause| cause.into_runtime_error());
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    let template = parse_output_template(OUTPUT_PROTOTYPE_CONTEXT, rest.view())?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&PLUS_LIKE_EXTENSION, BUILTIN_NAME)?;
    }
    let template =
        super::output_prototype::resolve_output_template(OUTPUT_PROTOTYPE_CONTEXT, &template)
            .await?;
    let unqualified_template = rest.is_empty();
    let branch_runtime = runtime.clone();
    super::preparation::execute_in_context(
        runtime.as_ref(),
        PLUS_CATALOG_ENTRY.identity,
        runmat_types::OperatorKind::Add,
        2 + rest.len(),
        lhs,
        rhs,
        |lhs, rhs| async move {
            let (base, host_branch) = match (lhs, rhs) {
                (Value::GpuTensor(la), Value::GpuTensor(lb)) => {
                    (plus_gpu_pair(branch_runtime.as_ref(), la, lb).await?, false)
                }
                (Value::GpuTensor(la), rhs) => (
                    plus_gpu_host_left(branch_runtime.as_ref(), la, rhs).await?,
                    false,
                ),
                (lhs, Value::GpuTensor(rb)) => (
                    plus_gpu_host_right(branch_runtime.as_ref(), lhs, rb).await?,
                    false,
                ),
                (lhs, rhs) => {
                    let typed_host_pair = matches!((&lhs, &rhs),
                        (Value::Tensor(left), Value::Tensor(right))
                            if left.numeric_dtype() == NumericDType::F32
                                && right.numeric_dtype() == NumericDType::F32);
                    (plus_host(lhs, rhs)?, typed_host_pair)
                }
            };
            publication::finish(
                base,
                &template,
                host_branch,
                unqualified_template,
                branch_runtime.as_ref(),
            )
            .await
        },
    )
    .await
}

#[cfg(test)]
use super::output_prototype::real_to_complex;
use super::output_prototype::{parse_output_template, OutputPrototypeContext, OutputTemplate};

const OUTPUT_PROTOTYPE_CONTEXT: OutputPrototypeContext = OutputPrototypeContext {
    identity: PLUS_CATALOG_ENTRY.identity,
    invalid_argument: &PLUS_ERROR_INVALID_ARGUMENT,
    invalid_input: &PLUS_ERROR_INVALID_INPUT,
};

mod provider;
use provider::{plus_gpu_host_left, plus_gpu_host_right, plus_gpu_pair};

mod publication;
mod text;

mod host;
use host::{is_real_integer_operand, plus_host};

#[cfg(test)]
mod tests;
