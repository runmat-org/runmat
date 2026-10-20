use super::super::typecast_builtin;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntValue, IntegerStorage, Tensor, Value};

#[test]
fn preserves_integer_bytes_width_and_orientation() {
    let input =
        Tensor::new_integer(IntegerStorage::U32(vec![1, 255, 256]), vec![1, 3]).expect("input");
    let Value::Tensor(output) = block_on(typecast_builtin(vec![
        Value::Tensor(input),
        Value::String("uint8".to_string().into()),
    ]))
    .expect("typecast") else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape(), &[1, 12]);
    let mut expected = Vec::new();
    for value in [1_u32, 255, 256] {
        expected.extend_from_slice(&value.to_ne_bytes());
    }
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(expected))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_inputs_preserve_typecast_vector_orientation_and_exact_bytes() {
    for (shape, expected_shape) in [(Vec::new(), vec![1, 2]), (vec![2], vec![4, 1])] {
        let values = if shape.is_empty() {
            vec![0x0102]
        } else {
            vec![0x0102, 0x0304]
        };
        let input = Tensor::new_integer(IntegerStorage::U16(values.clone()), shape).expect("input");
        let Value::Tensor(output) = block_on(typecast_builtin(vec![
            Value::Tensor(input),
            Value::String("uint8".into()),
        ]))
        .expect("typecast") else {
            panic!("expected byte tensor");
        };
        assert_eq!(output.shape(), expected_shape.as_slice());
        let expected = values
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(expected))
        );
    }
}

#[test]
fn round_trips_exact_uint64_bits() {
    let original = [0_u64, (1_u64 << 63) + 17, u64::MAX];
    let input =
        Tensor::new_integer(IntegerStorage::U64(original.to_vec()), vec![3, 1]).expect("input");
    let bytes = block_on(typecast_builtin(vec![
        Value::Tensor(input),
        Value::String("uint8".to_string().into()),
    ]))
    .expect("to bytes");
    let Value::Tensor(output) = block_on(typecast_builtin(vec![
        bytes,
        Value::String("uint64".to_string().into()),
    ]))
    .expect("from bytes") else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape(), &[3, 1]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(original.to_vec()))
    );
}

#[test]
fn like_complex_integer_groups_adjacent_components_exactly() {
    let input = Tensor::new_integer(
        IntegerStorage::I16(vec![-1, 2, i16::MIN, i16::MAX]),
        vec![1, 4],
    )
    .expect("input");
    let prototype = ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I16(vec![0]),
        )
        .expect("prototype storage"),
        vec![1, 1],
    )
    .expect("prototype");
    let Value::ComplexTensor(output) = block_on(typecast_builtin(vec![
        Value::Tensor(input),
        Value::String("like".to_string().into()),
        Value::ComplexTensor(prototype),
    ]))
    .expect("complex like") else {
        panic!("expected complex tensor");
    };
    assert_eq!(output.shape(), &[1, 2]);
    let storage = output.integer_storage().expect("integer complex");
    assert_eq!(storage.real(), &IntegerStorage::I16(vec![-1, i16::MIN]));
    assert_eq!(storage.imag(), &IntegerStorage::I16(vec![2, i16::MAX]));
}

#[test]
fn rejects_matrix_and_indivisible_byte_count() {
    let matrix =
        Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![2, 2]).expect("matrix");
    assert!(block_on(typecast_builtin(vec![
        Value::Tensor(matrix),
        Value::String("uint16".to_string().into()),
    ]))
    .is_err());
    assert!(block_on(typecast_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::String("uint16".to_string().into()),
    ]))
    .is_err());
}

#[test]
fn registered_binding_dispatches_exact_storage() {
    let input =
        Tensor::new_integer(IntegerStorage::I16(vec![-1, i16::MIN]), vec![1, 2]).expect("input");
    let result = crate::dispatcher::call_builtin(
        "typecast",
        &[
            Value::Tensor(input),
            Value::String("uint16".to_string().into()),
        ],
    )
    .expect("registered typecast");
    let Value::Tensor(output) = result.value() else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U16(vec![u16::MAX, 1_u16 << 15]))
    );
}
