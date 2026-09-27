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
    let protected: Vec<_> = borrowed.iter().collect();
    let _borrowed_metadata: Vec<_> = protected
        .iter()
        .map(|handle| crate::builtins::common::gpu_helpers::HandleMetadataRestoreGuard::new(handle))
        .collect();
    let converted = match value {
        Value::GpuTensor(handle) => single_from_gpu(handle, &protected).await?,
        host => host::convert(host)?,
    };
    let owned =
        crate::builtins::common::resident_output::ResidentResultGuard::new(&converted, &protected)
            .map_err(|detail| {
                single_error_with_detail(&runmat_builtins::SINGLE_ERROR_INTERNAL, detail)
            })?;
    let intermediate = match &converted {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let mut template_protected = protected.clone();
    if let Some(handle) = &intermediate {
        template_protected.push(handle);
    }
    let result = apply_output_template(converted, &template, &template_protected).await?;
    owned.publish(&result);
    Ok(result)
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
