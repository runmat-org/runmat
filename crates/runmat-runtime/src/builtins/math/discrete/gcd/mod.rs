//! MATLAB-compatible `gcd` execution.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::math::discrete::number_theory::binary::{
    binary_error, coefficient_value, extended_gcd, gcd, magnitude_value, resolve_output,
    BinaryInput, SameSizeOrScalarPlan, GCD_CONTEXT,
};
use crate::BuiltinResult;

#[runtime_builtin(
    name = "gcd",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::discrete::gcd"
)]
async fn gcd_builtin(left: Value, right: Value) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if matches!(requested, Some(count) if count > 3) {
        return Err(binary_error(
            &GCD_CONTEXT,
            GCD_CONTEXT.invalid,
            "at most three outputs are supported",
        ));
    }
    if requested == Some(0) {
        return Ok(Value::OutputList(Vec::new()));
    }

    let left = BinaryInput::from_value(left, &GCD_CONTEXT).await?;
    let right = BinaryInput::from_value(right, &GCD_CONTEXT).await?;
    let output = resolve_output(&left, &right, &GCD_CONTEXT)?;
    let plan = SameSizeOrScalarPlan::new(&left, &right, &GCD_CONTEXT)?;
    let extended = requested.is_some_and(|count| count >= 2);
    if extended
        && output
            .class
            .integer_class()
            .is_some_and(|integer| !integer.is_signed())
    {
        return Err(binary_error(
            &GCD_CONTEXT,
            GCD_CONTEXT.invalid,
            "Bézout coefficients require double, single, or signed integer inputs",
        ));
    }

    let mut divisors = Vec::with_capacity(plan.len());
    let mut first = extended.then(|| Vec::with_capacity(plan.len()));
    let mut second = extended.then(|| Vec::with_capacity(plan.len()));
    for (left_index, right_index) in plan.iter() {
        if extended {
            let (divisor, mut left_coefficient, mut right_coefficient) =
                extended_gcd(left.data[left_index], right.data[right_index]);
            if left.negative[left_index] {
                left_coefficient = -left_coefficient;
            }
            if right.negative[right_index] {
                right_coefficient = -right_coefficient;
            }
            divisors.push(divisor);
            first
                .as_mut()
                .expect("extended output")
                .push(left_coefficient);
            second
                .as_mut()
                .expect("extended output")
                .push(right_coefficient);
        } else {
            divisors.push(gcd(left.data[left_index], right.data[right_index]));
        }
    }
    assemble_outputs(
        requested,
        divisors,
        first,
        second,
        plan.output_shape,
        output,
    )
}

fn assemble_outputs(
    requested: Option<usize>,
    divisors: Vec<u128>,
    first: Option<Vec<i128>>,
    second: Option<Vec<i128>>,
    shape: Vec<usize>,
    output: crate::builtins::math::discrete::number_theory::binary::BinaryOutput,
) -> BuiltinResult<Value> {
    let divisor = magnitude_value(divisors, shape.clone(), output, &GCD_CONTEXT)?;
    let Some(count) = requested else {
        return Ok(divisor);
    };
    if count == 1 {
        return Ok(Value::OutputList(vec![divisor]));
    }
    let first = coefficient_value(
        first.expect("extended output"),
        shape.clone(),
        output,
        &GCD_CONTEXT,
    )?;
    if count == 2 {
        return Ok(Value::OutputList(vec![divisor, first]));
    }
    let second = coefficient_value(
        second.expect("extended output"),
        shape,
        output,
        &GCD_CONTEXT,
    )?;
    Ok(Value::OutputList(vec![divisor, first, second]))
}

#[cfg(test)]
mod tests;
