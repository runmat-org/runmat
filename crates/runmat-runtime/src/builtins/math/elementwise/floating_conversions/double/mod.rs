//! `double` runtime binding and conversion orchestration.

use runmat_builtins::{BuiltinErrorDescriptor, DOUBLE_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::errors::FloatingConversion;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
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
    context = true,
    builtin_path = "crate::builtins::math::elementwise::floating_conversions::double"
)]
async fn double_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    dispatch::execute(runtime.as_ref(), value, rest).await
}

mod dispatch;
mod host;
mod execution;
mod provider;
mod residency;
mod template;

use provider::*;
use template::*;

#[cfg(test)]
pub(crate) mod tests;
