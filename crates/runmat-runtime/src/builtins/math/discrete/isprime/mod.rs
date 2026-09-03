//! MATLAB-compatible `isprime` execution.

use runmat_builtins::{
    BuiltinErrorDescriptor, ISPRIME_ERROR_INTERNAL, ISPRIME_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

mod evaluation;
use evaluation::evaluate;

const BUILTIN_NAME: &str = "isprime";

#[derive(Clone, Copy)]
enum IsPrimeError {
    InvalidInput,
    Internal,
}

impl IsPrimeError {
    const fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::InvalidInput => &ISPRIME_ERROR_INVALID_INPUT,
            Self::Internal => &ISPRIME_ERROR_INTERNAL,
        }
    }
}

fn isprime_error(error: IsPrimeError, detail: impl std::fmt::Display) -> RuntimeError {
    let descriptor = error.descriptor();
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    if matches!(error, IsPrimeError::InvalidInput) {
        builder = builder.with_gpu_gather_retry(GpuGatherRetry::Never);
    }
    builder.build()
}

#[runtime_builtin(
    name = "isprime",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::discrete::isprime"
)]
async fn isprime_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(isprime_error(
            IsPrimeError::InvalidInput,
            "expected exactly one input argument",
        ));
    }
    evaluate(value)
}

#[cfg(test)]
mod tests;
