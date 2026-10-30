//! MATLAB-compatible `times` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::TIMES_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, TIMES_CATALOG_ENTRY, TIMES_ERROR_INTERNAL,
    TIMES_ERROR_INVALID_ARGUMENT, TIMES_ERROR_INVALID_INPUT, TIMES_ERROR_SIZE_MISMATCH,
    TIMES_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

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
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::times"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "times",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
        ProviderHook::Custom("scalar_mul"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses elem_mul for admitted real and complex-interleaved pairs; complex singleton expansion uses the provider broadcast hook directly, while real expansion can use repmat. Real scalar pairs can use scalar_mul; unsupported hooks fall back to host execution.",
};

const FUSION_SEMANTICS: crate::builtins::common::spec::FusionSemanticDeclaration =
    crate::builtins::common::spec::FusionSemanticDeclaration {
        binding: runmat_builtins::BuiltinBindingIdentity {
            builtin: TIMES_CATALOG_ENTRY.identity,
            variant: "default",
        },
        revision: 1,
        admit: None,
        admit_static: None,
        helpers: &[],
        required_real_unary: None,
        required_arithmetic: None,
        scalar_operation: Some(runmat_types::ScalarOperation::Operator(
            runmat_types::OperatorKind::ElementwiseMultiply,
        )),
    };

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::times",
    semantics = FUSION_SEMANTICS,
    host_resources = crate::builtins::math::host_resources::real_binary
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "times",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            crate::builtins::common::spec::write_real_binary_expression(
                runmat_types::OperatorKind::ElementwiseMultiply, ctx, writer,
            )
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain product; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "times";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn times_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn times_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "times",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::times"
)]
async fn times_builtin(
    runtime: Option<RuntimeContext>,
    lhs: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&lhs, &rhs].into_iter().chain(rest.iter()))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::TIMES_DESCRIPTOR,
        &runmat_builtins::TIMES_ERROR_TOO_MANY_OUTPUTS,
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
            &TIMES_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let template =
        super::output_prototype::resolve_output_template(OUTPUT_PROTOTYPE_CONTEXT, &template)
            .await?;
    let provider_runtime = runtime.clone();
    super::preparation::execute(
        runtime.as_ref(),
        TIMES_CATALOG_ENTRY.identity,
        runmat_types::OperatorKind::ElementwiseMultiply,
        2 + rest.len(),
        lhs,
        rhs,
        |lhs, rhs| async move {
            let base = match (lhs, rhs) {
                (Value::GpuTensor(la), Value::GpuTensor(lb)) => {
                    times_gpu_pair(provider_runtime.as_ref(), la, lb).await
                }
                (Value::GpuTensor(la), rhs) => {
                    times_gpu_host_left(provider_runtime.as_ref(), la, rhs).await
                }
                (lhs, Value::GpuTensor(rb)) => {
                    times_gpu_host_right(provider_runtime.as_ref(), lhs, rb).await
                }
                (lhs, rhs) => Ok(times_host(lhs, rhs)?),
            }?;
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
    identity: TIMES_CATALOG_ENTRY.identity,
    invalid_argument: &TIMES_ERROR_INVALID_ARGUMENT,
    invalid_input: &TIMES_ERROR_INVALID_INPUT,
};

mod provider;
use provider::{times_gpu_host_left, times_gpu_host_right, times_gpu_pair};

mod host;
pub(crate) use host::times_host;

#[cfg(test)]
mod tests;
