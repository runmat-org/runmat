use super::*;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor};

#[test]
fn empty_square_rectangular_and_nd_shapes_are_exact() {
    assert!(output(Vec::new(), &[0, 0]).data.is_empty());
    assert_eq!(output(vec![Value::Num(3.0)], &[3, 3]).data.len(), 9);
    assert_eq!(
        output(vec![Value::Num(2.0), Value::Num(4.0)], &[2, 4])
            .data
            .len(),
        8
    );
    assert_eq!(
        output(
            vec![Value::Num(2.0), Value::Num(3.0), Value::Num(4.0)],
            &[2, 3, 4]
        )
        .data
        .len(),
        24
    );
}

#[test]
fn integer_vectors_are_exact_and_trailing_singletons_normalize() {
    let sizes = Tensor::new_integer(IntegerStorage::U64(vec![2, 3, 1]), vec![1, 3]).unwrap();
    output(vec![Value::Tensor(sizes)], &[2, 3]);
    output(
        vec![Value::Int(IntValue::I64(-2)), Value::Int(IntValue::U8(3))],
        &[0, 3],
    );
}

#[test]
fn every_integer_storage_class_and_native_single_define_sizes() {
    let storages = [
        IntegerStorage::I8(vec![2, 3]),
        IntegerStorage::I16(vec![2, 3]),
        IntegerStorage::I32(vec![2, 3]),
        IntegerStorage::I64(vec![2, 3]),
        IntegerStorage::U8(vec![2, 3]),
        IntegerStorage::U16(vec![2, 3]),
        IntegerStorage::U32(vec![2, 3]),
        IntegerStorage::U64(vec![2, 3]),
    ];
    for storage in storages {
        let sizes = Tensor::new_integer(storage, vec![1, 2]).unwrap();
        output(vec![Value::Tensor(sizes)], &[2, 3]);
    }
    let sizes = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap();
    output(vec![Value::Tensor(sizes)], &[2, 3]);
    let scalar = Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap();
    output(vec![Value::Tensor(scalar)], &[4, 4]);
}

#[test]
fn invalid_size_representations_are_structured_errors() {
    assert_eq!(
        run(vec![Value::Num(1.5)]).unwrap_err().identifier(),
        Some("RunMat:cell:InvalidSize")
    );
    let column = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
    assert_eq!(
        run(vec![Value::Tensor(column)]).unwrap_err().identifier(),
        Some("RunMat:cell:InvalidSize")
    );
    assert_eq!(
        run(vec![Value::Bool(true)]).unwrap_err().identifier(),
        Some("RunMat:cell:InvalidInput")
    );
    let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
    assert_eq!(
        run(vec![Value::LogicalArray(logical)])
            .unwrap_err()
            .identifier(),
        Some("RunMat:cell:InvalidInput")
    );
}

#[test]
fn compatible_elements_are_independent_empty_double_arrays() {
    let cells = output(vec![Value::Num(2.0)], &[2, 2]);
    for value in cells.data {
        let Value::Tensor(value) = value else {
            panic!("expected double empty")
        };
        assert_eq!(value.shape, vec![0, 0]);
        assert!(value.is_empty());
    }
}
