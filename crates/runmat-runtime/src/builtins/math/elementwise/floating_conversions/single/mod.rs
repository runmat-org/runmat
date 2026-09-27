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
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::SINGLE_CATALOG_ENTRY.descriptor,
        &runmat_builtins::SINGLE_ERROR_INVALID_ARGUMENT,
        CONVERSION.name(),
    )?;
    let template = parse_output_template(&rest)?;
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINGLE_LIKE_OUTPUT_EXTENSION,
            CONVERSION.name(),
        )?;
    }
    template.validate()?;
    template.validate_input(&value)?;
    let borrowed: Vec<_> = std::iter::once(&value)
        .chain(rest.iter())
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle.clone())
            } else {
                None
            }
        })
        .collect();
    let signature = usize::from(!rest.is_empty());
    let input_count = 1 + rest.len();
    super::preparation::execute(
        runmat_builtins::SINGLE_CATALOG_ENTRY.identity,
        value,
        signature,
        input_count,
        |value| execution::convert(value, template, borrowed),
    )
    .await
}

mod host;
mod execution;
mod provider;
mod residency;
mod storage;
mod template;

use provider::*;
use template::*;

#[cfg(test)]
pub(crate) mod tests;
