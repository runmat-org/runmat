//! `double` runtime binding and conversion orchestration.

use runmat_builtins::{BuiltinErrorDescriptor, DOUBLE_ERROR_INVALID_INPUT, DOUBLE_LIKE_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::errors::FloatingConversion;
use crate::{BuiltinResult, RuntimeError};

mod specs;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
const CONVERSION: FloatingConversion = FloatingConversion::Double;

fn double_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    CONVERSION.error_with_detail(error, detail)
}

fn conversion_error(type_name: &str) -> RuntimeError {
    CONVERSION.conversion_error(&DOUBLE_ERROR_INVALID_INPUT, type_name)
}

#[runtime_builtin(
    name = "double",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::floating_conversions::double"
)]
async fn double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    if matches!(template, OutputTemplate::Like(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOUBLE_LIKE_EXTENSION,
            CONVERSION.name(),
        )?;
    }
    let converted = match value {
        Value::GpuTensor(handle) => double_from_gpu(handle).await?,
        host => host::convert(host)?,
    };
    apply_output_template(converted, &template).await
}

mod host;
mod provider;
mod residency;
mod template;

use provider::*;
use template::*;

#[cfg(test)]
pub(crate) mod tests;
