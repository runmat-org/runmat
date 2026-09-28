//! Complex conjugation.

mod host;
mod provider;
mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_builtins::{
    CONJ_CHARACTER_INPUT_EXTENSION, CONJ_ERROR_INTERNAL, CONJ_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;
const BUILTIN_NAME: &str = "conj";

mod errors;
use errors::builtin_error_with_detail;

#[runtime_builtin(
    name = "conj",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::complex_components::conj"
)]
async fn conj_builtin(value: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::CONJ_DESCRIPTOR,
        &runmat_builtins::CONJ_ERROR_TOO_MANY_OUTPUTS,
        "conj",
    )?;
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONJ_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    super::preparation::execute(
        runmat_builtins::CONJ_CATALOG_ENTRY.identity,
        value,
        |value| async move {
            match value {
                Value::GpuTensor(handle) => provider::execute(handle).await,
                host_value => host::execute(host_value),
            }
        },
    )
    .await
}

pub(crate) use host::conjugate_integer_imaginary_storage;

#[cfg(all(test, feature = "wgpu"))]
mod test_helpers;
#[cfg(all(test, feature = "wgpu"))]
use test_helpers::{conj_gpu, conj_real};

#[cfg(test)]
mod tests;
