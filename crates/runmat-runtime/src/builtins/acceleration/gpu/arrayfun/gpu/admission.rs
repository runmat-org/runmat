use super::super::error::arrayfun_flow;
use super::super::input::normalized_shape;
use crate::builtins::common::broadcast::broadcast_shapes;
use crate::builtins::math::elementwise::integer_arithmetic::validate_integer_binary_inputs;
use crate::BuiltinResult;
use runmat_builtins::ArrayfunGpuCallback;
use runmat_value::Value;

fn operand_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::GpuTensor(value) => Some(value.shape()),
        Value::Tensor(value) => Some(value.shape()),
        Value::ComplexTensor(value) => Some(value.shape()),
        Value::LogicalArray(value) => Some(&value.shape),
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) | Value::Bool(_) => Some(&[1, 1]),
        _ => None,
    }
}

pub(super) fn shape(
    inputs: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<Vec<usize>>> {
    let Some(shapes) = inputs
        .iter()
        .map(|value| operand_shape(value).map(normalized_shape))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let Some((first, rest)) = shapes.split_first() else {
        return Ok(None);
    };
    let shape = rest.iter().try_fold(first.to_vec(), |shape, next| {
        broadcast_shapes("arrayfun", &shape, next).map_err(arrayfun_flow)
    })?;
    if !shape.contains(&0) {
        shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| arrayfun_flow("arrayfun: output shape exceeds addressable storage"))?;
    }
    Ok(Some(shape))
}

fn scalar(value: &Value) -> bool {
    operand_shape(value).is_some_and(|shape| !shape.contains(&0) && shape.iter().all(|&d| d == 1))
}

pub(super) fn equivalent(
    operation: ArrayfunGpuCallback,
    name: &str,
    inputs: crate::call::FunctionArgumentView<'_>,
) -> bool {
    match (operation, inputs.len()) {
        (ArrayfunGpuCallback::Unary(_), 1) => true,
        (ArrayfunGpuCallback::Binary(_), 2) => {
            let (left, right) = (&inputs[0], &inputs[1]);
            // Scalar callback admission and whole-array admission must both hold.
            // A varying double beside an integer is not a scalar double merely
            // because each mapped invocation receives a scalar.
            validate_integer_binary_inputs(left, right, true, true, name).is_ok()
                && validate_integer_binary_inputs(left, right, scalar(left), scalar(right), name)
                    .is_ok()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests;
