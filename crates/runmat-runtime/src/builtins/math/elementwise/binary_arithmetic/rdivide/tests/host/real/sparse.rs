use super::super::super::*;
use runmat_value::{NumericDType, SparseTensor};

#[test]
fn rdivide_sparse_scalar_preserves_csc_and_class() {
    let sparse = SparseTensor::new_f32(3, 2, vec![0, 1, 2], vec![1, 0], vec![6.0, -8.0]).unwrap();
    for divisor in [2.0, -2.0, f64::INFINITY] {
        let Value::SparseTensor(output) = rdivide_builtin(
            Value::SparseTensor(sparse.clone()),
            Value::Num(divisor),
            Vec::new(),
        )
        .expect("sparse scalar division") else {
            panic!("expected sparse result")
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        if divisor.is_infinite() {
            assert_eq!(output.nnz(), 0);
            assert_eq!(&*output.col_ptrs, &[0, 0, 0]);
        } else {
            assert_eq!(&*output.col_ptrs, &[0, 1, 2]);
            assert_eq!(&*output.row_indices, &[1, 0]);
            assert_eq!(output.as_f32_slice().unwrap()[0], 6.0_f32 / divisor as f32);
        }
    }
}

#[test]
fn rdivide_sparse_zero_divisor_materializes_implicit_nan() {
    let sparse = SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![4.0]).unwrap();
    let Value::Tensor(output) =
        rdivide_builtin(Value::SparseTensor(sparse), Value::Num(0.0), Vec::new())
            .expect("full quotient")
    else {
        panic!("expected dense result")
    };
    assert!(output.materialize_f64()[1].is_nan());
}

#[test]
fn rdivide_sparse_one_entry_preserves_complex_and_integer_classes() {
    let complex =
        SparseTensor::new_complex_f32(2, 1, vec![0, 1], vec![0], vec![(4.0, 2.0)]).unwrap();
    let Value::SparseTensor(complex) = rdivide_builtin(
        Value::SparseTensor(complex),
        Value::Complex(1.0, 1.0),
        Vec::new(),
    )
    .expect("complex sparse quotient") else {
        panic!("expected sparse complex quotient")
    };
    assert_eq!(complex.numeric_dtype(), Some(NumericDType::F32));
    assert_eq!(
        complex.as_complex_f32_slice(),
        Some(&[runmat_value::ComplexElement(3.0, -1.0)][..])
    );
    assert_eq!(&*complex.row_indices, &[0]);

    let integer = SparseTensor::new_integer(
        2,
        1,
        vec![0, 1],
        vec![1],
        IntegerStorage::U64(vec![u64::MAX]),
    )
    .unwrap();
    let Value::SparseTensor(integer) =
        rdivide_builtin(Value::SparseTensor(integer), Value::Num(2.0), Vec::new())
            .expect("integer sparse quotient")
    else {
        panic!("expected sparse integer quotient")
    };
    assert_eq!(
        integer.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 63]))
    );
    assert_eq!(&*integer.row_indices, &[1]);
}
