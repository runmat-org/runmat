//! Complex conjugation.

mod host;
pub(crate) use host::execute as conjugate_host_value;
mod admission;
mod provider;
mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_builtins::{CONJ_ERROR_INTERNAL, CONJ_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
const BUILTIN_NAME: &str = "conj";

mod errors;
use errors::builtin_error_with_detail;

#[runtime_builtin(
    name = "conj",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::complex_components::conj"
)]
async fn conj_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::CONJ_DESCRIPTOR,
        &runmat_builtins::CONJ_ERROR_TOO_MANY_OUTPUTS,
        "conj",
    )?;
    admission::ensure_character_enabled(&value)?;
    super::preparation::execute(
        runtime.as_ref(),
        runmat_builtins::CONJ_CATALOG_ENTRY.identity,
        value,
        |value| async move {
            let conjugated = match value {
                Value::GpuTensor(handle) => provider::execute(handle).await,
                host_value => host::execute(host_value),
            }?;
            RootedValue::new(conjugated).map_err(|cause| cause.into_runtime_error())
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
