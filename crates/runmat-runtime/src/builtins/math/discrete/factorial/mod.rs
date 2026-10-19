//! MATLAB-compatible factorial execution.

use runmat_builtins::{
    BuiltinErrorDescriptor, FACTORIAL_ERROR_GPU_UNSUPPORTED, FACTORIAL_ERROR_INTERNAL,
    FACTORIAL_ERROR_INVALID_ARGUMENT, FACTORIAL_ERROR_INVALID_INPUT,
    FACTORIAL_ERROR_TOO_MANY_OUTPUTS, FACTORIAL_LOGICAL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

mod arguments;
mod evaluation;
mod integer;
mod provider;
pub(crate) mod spec;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

use arguments::{apply_output_template, parse_output_template};
use evaluation::factorial_tensor;
use provider::factorial_gpu;

const BUILTIN_NAME: &str = "factorial";

#[derive(Clone, Copy)]
enum FactorialError {
    InvalidArgument,
    InvalidInput,
    GpuUnsupported,
    Internal,
    TooManyOutputs,
}

impl FactorialError {
    const fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::InvalidArgument => &FACTORIAL_ERROR_INVALID_ARGUMENT,
            Self::InvalidInput => &FACTORIAL_ERROR_INVALID_INPUT,
            Self::GpuUnsupported => &FACTORIAL_ERROR_GPU_UNSUPPORTED,
            Self::Internal => &FACTORIAL_ERROR_INTERNAL,
            Self::TooManyOutputs => &FACTORIAL_ERROR_TOO_MANY_OUTPUTS,
        }
    }

    const fn gather_retry(self) -> Option<GpuGatherRetry> {
        match self {
            Self::InvalidInput | Self::GpuUnsupported => Some(GpuGatherRetry::Never),
            Self::InvalidArgument | Self::Internal | Self::TooManyOutputs => None,
        }
    }
}

fn factorial_error(error: FactorialError, detail: impl std::fmt::Display) -> RuntimeError {
    let descriptor = error.descriptor();
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(policy) = error.gather_retry() {
        builder = builder.with_gpu_gather_retry(policy);
    }
    builder.build()
}

#[runtime_builtin(
    name = "factorial",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::discrete::factorial"
)]
async fn factorial_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    let output = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => factorial_gpu(handle).await?,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(factorial_error(
                FactorialError::InvalidInput,
                "complex inputs are not supported; use gamma(z + 1) instead",
            ));
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            return Err(factorial_error(
                FactorialError::InvalidInput,
                "expected numeric or logical input",
            ));
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FACTORIAL_LOGICAL_EXTENSION,
                BUILTIN_NAME,
            )?;
            evaluate_host(value)?
        }
        other => evaluate_host(other)?,
    };
    apply_output_template(base, &output).await
}

fn evaluate_host(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|error| factorial_error(FactorialError::InvalidInput, error))?;
    factorial_tensor(tensor).map(tensor::tensor_into_value)
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(factorial_error(
            FactorialError::TooManyOutputs,
            "only one output is defined",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
