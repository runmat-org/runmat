//! MATLAB-compatible `diff` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, DIFF_CHARACTER_INPUT_EXTENSION, DIFF_DESCRIPTOR,
    DIFF_EMPTY_CONTROL_EXTENSION, DIFF_ERROR_INTERNAL, DIFF_ERROR_INVALID_ARGUMENT,
    DIFF_ERROR_INVALID_INPUT, DIFF_ERROR_TOO_MANY_OUTPUTS, DIFF_ZERO_ORDER_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, NumericStorage, Tensor, Value,
};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::same_class_saturating_subtract;
use crate::builtins::math::symbolic::{
    symbolic_expr_to_value, symbolic_variable_name_from_value, value_to_symbolic_scalar,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod arguments;
mod geometry;
use arguments::*;
pub(in crate::builtins::math) use geometry::{difference_geometry, DifferenceGeometry};
mod provider;
use provider::*;
mod host;
pub use host::diff_tensor_host;
use host::*;
const NAME: &str = "diff";

#[cfg(test)]
#[path = "diff/owning_tests.rs"]
mod owning_tests;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::diff")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diff",
    op_kind: GpuOpKind::Custom("finite-difference"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("diff_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Eligible real floating input uses its owning provider's diff_dim hook. Unsupported automatic placement may take an admitted native-preserving host route before an attempt; unsupported explicit input is not gathered and attempted errors are terminal.",
};

fn diff_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn diff_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    diff_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn diff_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    diff_descriptor_error_with_detail(&DIFF_ERROR_INVALID_ARGUMENT, detail)
}

fn diff_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    diff_descriptor_error_with_detail(&DIFF_ERROR_INVALID_INPUT, detail)
}

fn diff_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    diff_descriptor_error_with_detail(&DIFF_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::diff")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diff",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently delegates to the runtime implementation; providers can override with custom kernels.",
};

#[runtime_builtin(
    name = "diff",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::diff"
)]
async fn diff_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &DIFF_DESCRIPTOR,
        &DIFF_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )
    .map_err(super::numerical_execution::terminal)?;
    let call = super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::Difference,
        runtime.as_ref(),
    )?;
    if rest.len() > 2 {
        return Err(diff_invalid_argument("diff: unsupported arguments"));
    }
    if let Value::Symbolic(expr) = value {
        call.host()?;
        return call.finish_value(diff_symbolic(expr, rest.view()));
    }
    if !matches!(
        value,
        Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(..)
            | Value::ComplexTensor(_)
            | Value::CharArray(_)
            | Value::GpuTensor(_)
    ) {
        return Err(diff_invalid_input(
            "expected numeric, logical, symbolic, or admitted character input",
        ));
    }

    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIFF_CHARACTER_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if rest.iter().any(is_empty_array) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIFF_EMPTY_CONTROL_EXTENSION,
            NAME,
        )?;
    }
    let (order, dim) = parse_arguments(rest.view())?;
    if order == 0 {
        crate::compatibility::ensure_builtin_extension_enabled(&DIFF_ZERO_ORDER_EXTENSION, NAME)?;
        return call.finish_value(Ok(value));
    }

    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(diff_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }

    if !matches!(value, Value::GpuTensor(_)) {
        call.host()?;
    }
    if let Value::GpuTensor(handle) = &value {
        gpu_helpers::require_resident_compute_owner(handle, || {
            diff_internal_error("diff: GPU input has no owning provider")
        })?;
    }
    let result = match value {
        Value::Tensor(tensor) => {
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(logical) => diff_logical_array(logical, order, dim),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor =
                tensor::value_into_tensor_for("diff", value).map_err(diff_invalid_input)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(diff_invalid_input)?;
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::ComplexTensor(tensor) => {
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::CharArray(chars) => diff_char_array(chars, order, dim),
        Value::GpuTensor(handle) => diff_gpu(handle, order, dim, &call).await,
        other => Err(diff_invalid_input(format!(
            "diff: unsupported input type {:?}; expected numeric, logical, or character data",
            other
        ))),
    };
    call.finish_value(result)
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&dim| dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, val| acc.saturating_mul(val))
}

#[cfg(test)]
pub(crate) mod tests;
