//! Complex conjugation.

mod host;
mod provider;
mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_builtins::{
    BuiltinErrorDescriptor, CONJ_CHARACTER_INPUT_EXTENSION, CONJ_ERROR_INTERNAL,
    CONJ_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "conj";

fn builtin_error_with_detail(
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
    name = "conj",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::complex_components::conj"
)]
async fn conj_builtin(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONJ_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match value {
        Value::GpuTensor(handle) => provider::execute(handle).await,
        host_value => host::execute(host_value),
    }
}

pub(crate) use host::conjugate_integer_imaginary_storage;

#[cfg(all(test, feature = "wgpu"))]
fn conj_real(value: Value) -> BuiltinResult<Value> {
    host::execute(value)
}

#[cfg(all(test, feature = "wgpu"))]
async fn conj_gpu(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    provider::execute(handle).await
}

#[cfg(test)]
mod tests;
