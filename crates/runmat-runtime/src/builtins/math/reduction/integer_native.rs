//! Exact native integer reductions shared by MATLAB reduction builtins.

use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

/// Reduces an integer tensor by saturated native addition along zero-based
/// dimensions. Callers select this only for MATLAB's explicit `"native"`
/// output mode; default reductions intentionally retain their double output.
pub(crate) fn sum(
    storage: &IntegerStorage,
    shape: &[usize],
    reduced_dims: &[usize],
) -> Result<Value, String> {
    if reduced_dims.is_empty() {
        return integer_storage_into_value(storage.clone(), shape.to_vec());
    }

    let shape = normalized_shape(shape);
    let mut output_shape = shape.clone();
    for &dim in reduced_dims {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }
    let output_len = element_count(&output_shape);
    let mut coords = vec![0usize; shape.len()];
    let mut output_coords = vec![0usize; shape.len()];
    let mut reduced = vec![false; shape.len()];
    for &dim in reduced_dims {
        if dim < reduced.len() {
            reduced[dim] = true;
        }
    }

    macro_rules! reduce {
        ($values:expr, $variant:ident) => {{
            let mut output = vec![0; output_len];
            for (linear, &value) in $values.iter().enumerate() {
                linear_to_multi(linear, &shape, &mut coords);
                for (dim, &coord) in coords.iter().enumerate() {
                    output_coords[dim] = if reduced[dim] { 0 } else { coord };
                }
                let output_index = multi_to_linear(&output_coords, &output_shape);
                output[output_index] = output[output_index].saturating_add(value);
            }
            IntegerStorage::$variant(output)
        }};
    }

    let output = match storage {
        IntegerStorage::I8(values) => reduce!(values, I8),
        IntegerStorage::I16(values) => reduce!(values, I16),
        IntegerStorage::I32(values) => reduce!(values, I32),
        IntegerStorage::I64(values) => reduce!(values, I64),
        IntegerStorage::U8(values) => reduce!(values, U8),
        IntegerStorage::U16(values) => reduce!(values, U16),
        IntegerStorage::U32(values) => reduce!(values, U32),
        IntegerStorage::U64(values) => reduce!(values, U64),
    };
    integer_storage_into_value(output, output_shape)
}

pub(crate) fn storage_from_scalar(value: &IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![*value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![*value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![*value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![*value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![*value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![*value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![*value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![*value]),
    }
}

fn integer_storage_into_value(storage: IntegerStorage, shape: Vec<usize>) -> Result<Value, String> {
    if storage.len() == 1 {
        return Ok(Value::Int(storage_value(&storage, 0)));
    }
    Ok(Value::Tensor(Tensor::new_integer(storage, shape)?))
}

fn storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn normalized_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn linear_to_multi(index: usize, shape: &[usize], output: &mut [usize]) {
    let mut remainder = index;
    for (dim, &size) in shape.iter().enumerate() {
        output[dim] = if size == 0 { 0 } else { remainder % size };
        if size != 0 {
            remainder /= size;
        }
    }
}

fn multi_to_linear(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (dim, &size) in shape.iter().enumerate() {
        index += coords[dim] * stride;
        stride *= size;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_preserves_uint64_bits_and_saturates() {
        let result =
            sum(&IntegerStorage::U64(vec![u64::MAX, 1, 5, 7]), &[2, 2], &[0]).expect("native sum");
        assert_eq!(
            result,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 12]), vec![1, 2])
                    .expect("expected tensor")
            )
        );
    }

    #[test]
    fn sum_handles_multidimensional_reduction_and_out_of_bounds_dim() {
        let storage = IntegerStorage::I16(vec![10, 20, 30, 40]);
        assert_eq!(
            sum(&storage, &[2, 2], &[1]).expect("row reduction"),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![40, 60]), vec![2, 1])
                    .expect("expected tensor")
            )
        );
        assert_eq!(
            sum(&storage, &[2, 2], &[]).expect("out of bounds"),
            Value::Tensor(Tensor::new_integer(storage, vec![2, 2]).expect("unchanged tensor"))
        );
    }
}
