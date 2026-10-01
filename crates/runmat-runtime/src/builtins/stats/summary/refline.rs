//! Public refline binding over the shared reference-line graphics service.

use crate::BuiltinResult;
#[cfg(test)]
use crate::RuntimeError;
use runmat_builtins::ReferenceLineOwner;
#[cfg(test)]
use runmat_builtins::{REFLINE_ERROR_INTERNAL, REFLINE_ERROR_INVALID_ARGUMENT};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(test)]
fn internal_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin("refline");
    if let Some(identifier) = REFLINE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "refline",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::refline"
)]
pub(crate) async fn refline_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::reference_lines::run_reference_lines(args, ReferenceLineOwner::Refline).await
}

#[cfg(test)]
mod tests;
