//! MATLAB-compatible `factor` execution.

use runmat_builtins::{BuiltinErrorDescriptor, FACTOR_ERROR_INTERNAL, FACTOR_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

mod arguments;
mod evaluation;

use arguments::parse_input;
use evaluation::result_value;

const BUILTIN_NAME: &str = "factor";

#[derive(Clone, Copy)]
enum FactorError {
    InvalidInput,
    Internal,
}

impl FactorError {
    const fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::InvalidInput => &FACTOR_ERROR_INVALID_INPUT,
            Self::Internal => &FACTOR_ERROR_INTERNAL,
        }
    }
}

fn factor_error(error: FactorError, detail: impl std::fmt::Display) -> RuntimeError {
    let descriptor = error.descriptor();
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    if matches!(error, FactorError::InvalidInput) {
        builder = builder.with_gpu_gather_retry(GpuGatherRetry::Never);
    }
    builder.build()
}

#[runtime_builtin(
    name = "factor",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::discrete::factor"
)]
async fn factor_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(factor_error(
            FactorError::InvalidInput,
            "expected exactly one input argument",
        ));
    }
    let input = parse_input(value)?;
    result_value(input)
}

#[cfg(test)]
mod tests;
