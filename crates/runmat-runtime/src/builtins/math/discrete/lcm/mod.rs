//! MATLAB-compatible `lcm` execution.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::math::discrete::number_theory::binary::{
    lcm, magnitude_value, resolve_output, BinaryInput, SameSizeOrScalarPlan, LCM_CONTEXT,
};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "lcm",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::discrete::lcm"
)]
async fn lcm_builtin(left: Value, right: Value) -> BuiltinResult<Value> {
    let left = BinaryInput::from_value(left, &LCM_CONTEXT).await?;
    let right = BinaryInput::from_value(right, &LCM_CONTEXT).await?;
    let output = resolve_output(&left, &right, &LCM_CONTEXT)?;
    let plan = SameSizeOrScalarPlan::new(&left, &right, &LCM_CONTEXT)?;
    let values = plan
        .iter()
        .map(|(left_index, right_index)| lcm(left.data[left_index], right.data[right_index]))
        .collect();
    magnitude_value(values, plan.output_shape, output, &LCM_CONTEXT)
}

#[cfg(test)]
mod tests;
