//! Range scaling for real numeric and logical arrays.

mod arguments;
mod broadcast;
mod compute;
mod defaults;
mod error;
mod operands;
mod provider;
mod spec;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::__runmat_wasm_register_gpu_spec_GPU_SPEC;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "rescale";

#[runtime_builtin(
    name = "rescale",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::rescale"
)]
async fn rescale_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let raw = arguments::parse(&rest)?;
    operands::ensure_integer_boundary(&value, "input").await?;
    operands::ensure_integer_boundary(&raw.lower, "lower bound").await?;
    operands::ensure_integer_boundary(&raw.upper, "upper bound").await?;
    if let Some(bound) = raw.input_min.as_ref() {
        operands::ensure_integer_boundary(bound, "InputMin").await?;
    }
    if let Some(bound) = raw.input_max.as_ref() {
        operands::ensure_integer_boundary(bound, "InputMax").await?;
    }

    let input = operands::input(value).await?;
    let lower = operands::bound(raw.lower, "lower bound").await?;
    let upper = operands::bound(raw.upper, "upper bound").await?;
    let input_min = match raw.input_min {
        Some(value) => operands::bound(value, "InputMin").await?,
        None => operands::BoundOperand::host(defaults::scalar_tensor(defaults::input_min(
            &input.tensor,
        ))),
    };
    let input_max = match raw.input_max {
        Some(value) => operands::bound(value, "InputMax").await?,
        None => operands::BoundOperand::host(defaults::scalar_tensor(defaults::input_max(
            &input.tensor,
        ))),
    };
    compute::rescale(input, lower, upper, input_min, input_max)
}
