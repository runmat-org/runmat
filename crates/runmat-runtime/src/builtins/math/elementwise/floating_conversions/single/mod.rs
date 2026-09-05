//! `single` runtime binding and conversion orchestration.

use runmat_builtins::{
    BuiltinErrorDescriptor, SINGLE_ERROR_INVALID_INPUT, SINGLE_LIKE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::errors::FloatingConversion;
use crate::{BuiltinResult, RuntimeError};

mod specs;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
const CONVERSION: FloatingConversion = FloatingConversion::Single;

fn single_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    CONVERSION.error_with_detail(error, detail)
}

fn conversion_error(type_name: &str) -> RuntimeError {
    CONVERSION.conversion_error(&SINGLE_ERROR_INVALID_INPUT, type_name)
}

#[runtime_builtin(
    name = "single",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::floating_conversions::single"
)]
async fn single_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINGLE_LIKE_OUTPUT_EXTENSION,
            CONVERSION.name(),
        )?;
    }
    let converted = match value {
        Value::GpuTensor(handle) => single_from_gpu(handle).await?,
        host => host::convert(host)?,
    };
    apply_output_template(converted, &template).await
}

mod host;
mod provider;
mod residency;
mod storage;
mod template;

use provider::*;
use template::*;

#[cfg(test)]
pub(crate) mod tests;
