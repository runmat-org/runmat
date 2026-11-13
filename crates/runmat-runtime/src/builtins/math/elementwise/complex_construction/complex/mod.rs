//! Class-aware construction of complex values.

mod host;
mod provider;
mod preparation;
mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_builtins::{BuiltinErrorDescriptor, COMPLEX_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) const BUILTIN_NAME: &str = "complex";

pub(super) fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "complex",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::complex_construction::complex"
)]
async fn complex_builtin(
    runtime: Option<RuntimeContext>,
    real: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::COMPLEX_DESCRIPTOR,
        &runmat_builtins::COMPLEX_ERROR_TOO_MANY_OUTPUTS,
        "complex",
    )?;
    match (rest.len(), rest.first()) {
        (0, _) => preparation::unary(runtime.as_ref(), real).await,
        (1, Some(imaginary)) => {
            preparation::binary(runtime.as_ref(), real, imaginary.clone()).await
        }
        _ => Err(error(
            &COMPLEX_ERROR_INVALID_ARGUMENT,
            format!("expected 1 or 2 input arguments, got {}", rest.len() + 1),
        )),
    }
}

#[cfg(test)]
mod tests;
