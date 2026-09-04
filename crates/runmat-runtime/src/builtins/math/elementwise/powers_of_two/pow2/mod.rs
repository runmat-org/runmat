mod binary;
mod compatibility;
mod errors;
mod input;
mod numeric;
mod provider;
mod specs;
mod unary;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

const BUILTIN_NAME: &str = "pow2";

#[runtime_builtin(
    name = "pow2",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::powers_of_two::pow2"
)]
async fn pow2_builtin(first: Value, mut rest: Vec<Value>) -> BuiltinResult<Value> {
    errors::reject_excess_outputs()?;
    match rest.pop() {
        None => {
            compatibility::validate_unary(&first).await?;
            unary::evaluate(first).await
        }
        Some(exponent) if rest.is_empty() => {
            compatibility::validate_binary(&first, &exponent).await?;
            binary::evaluate(first, exponent).await
        }
        Some(_) => Err(errors::invalid_argument("expected one or two inputs")),
    }
}
