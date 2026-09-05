mod errors;
mod extensions;
mod host;
mod provider;
pub(crate) mod specification;

use runmat_macros::runtime_builtin;
use runmat_value::{SymbolicFunction, Value};

use crate::builtins::math::symbolic::symbolic_function;
use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "heaviside";

#[runtime_builtin(
    name = "heaviside",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::heaviside"
)]
async fn heaviside_builtin(value: Value) -> BuiltinResult<Value> {
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Heaviside) {
        return Ok(symbolic);
    }
    extensions::validate(&value)?;
    match value {
        Value::GpuTensor(handle) => provider::execute(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(errors::invalid_input(
            "complex inputs are not supported for the real Heaviside step function",
        )),
        Value::String(_) | Value::StringArray(_) => Err(errors::invalid_input(
            "expected real numeric, logical, or character input",
        )),
        value => host::execute(value),
    }
}

#[cfg(test)]
mod tests;
