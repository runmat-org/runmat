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
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{
    reject_integer_logical_operands, try_integer_binary, IntegerBinaryOp,
};
use crate::builtins::math::elementwise::sparse::{try_sparse_binary, SparseBinaryOp};
use crate::builtins::math::elementwise::sparse_integer::try_typed_sparse_integer_binary;
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
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
        "Uses elem_mul for shape-compatible gpuArrays, including complex-interleaved handles, attempts provider-side implicit expansion with repmat, and uses scalar_mul when one operand is a real scalar; falls back to host execution for unsupported operand kinds.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::times"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "times",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({lhs} * {rhs})"))
        },
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
    builtin_path = "crate::builtins::math::elementwise::binary_arithmetic::times"
)]
async fn times_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
            &TIMES_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => times_gpu_pair(la, lb).await,
        (Value::GpuTensor(la), rhs) => times_gpu_host_left(la, rhs).await,
        (lhs, Value::GpuTensor(rb)) => times_gpu_host_right(lhs, rb).await,
        (lhs, rhs) => Ok(times_host(lhs, rhs)?),
    }?;
    apply_output_template(OUTPUT_PROTOTYPE_CONTEXT, base, &template).await
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
