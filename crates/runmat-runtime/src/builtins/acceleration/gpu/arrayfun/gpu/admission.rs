use super::super::error::arrayfun_flow;
use crate::builtins::common::broadcast::broadcast_shapes;
use crate::builtins::math::elementwise::integer_arithmetic::validate_integer_binary_inputs;
use crate::BuiltinResult;
use runmat_builtins::ArrayfunGpuCallback;
use runmat_value::Value;

fn operand_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::GpuTensor(value) => Some(value.shape()),
        Value::Tensor(value) => Some(&value.shape),
        Value::ComplexTensor(value) => Some(&value.shape),
        Value::LogicalArray(value) => Some(&value.shape),
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) | Value::Bool(_) => Some(&[1, 1]),
        _ => None,
    }
}

pub(super) fn shape(inputs: &[Value]) -> BuiltinResult<Option<Vec<usize>>> {
    let Some(shapes) = inputs.iter().map(operand_shape).collect::<Option<Vec<_>>>() else {
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

pub(super) fn equivalent(operation: ArrayfunGpuCallback, name: &str, inputs: &[Value]) -> bool {
    match (operation, inputs) {
        (ArrayfunGpuCallback::Unary(_), [_]) => true,
        (ArrayfunGpuCallback::Binary(_), [left, right]) => {
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
mod tests {
    use super::*;
    use runmat_builtins::ArrayfunGpuBinary;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn varying_double_integer_pair_is_not_whole_array_equivalent() {
        let integer = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, 7]),
                vec![1, 2],
            )
            .unwrap(),
        );
        let double = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
        for operation in [
            ArrayfunGpuBinary::Add,
            ArrayfunGpuBinary::Subtract,
            ArrayfunGpuBinary::Multiply,
            ArrayfunGpuBinary::RightDivide,
            ArrayfunGpuBinary::LeftDivide,
        ] {
            let operation = ArrayfunGpuCallback::Binary(operation);
            assert!(!equivalent(
                operation,
                "mapped",
                &[integer.clone(), double.clone()]
            ));
            assert!(!equivalent(
                operation,
                "mapped",
                &[double.clone(), integer.clone()]
            ));
            assert!(equivalent(
                operation,
                "mapped",
                &[integer.clone(), Value::Num(2.0)]
            ));
            assert!(equivalent(
                operation,
                "mapped",
                &[integer.clone(), integer.clone()]
            ));
        }
    }

    #[test]
    fn mixed_float_and_broadcast_shapes_remain_admitted() {
        let single = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let double = Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap());
        assert!(equivalent(
            ArrayfunGpuCallback::Binary(ArrayfunGpuBinary::Add),
            "plus",
            &[single.clone(), double.clone()]
        ));
        assert_eq!(shape(&[single, double]).unwrap(), Some(vec![2, 3]));
    }
}
