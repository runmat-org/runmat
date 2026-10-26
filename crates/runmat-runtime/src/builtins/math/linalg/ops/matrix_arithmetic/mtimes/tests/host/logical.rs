use super::*;

pub(super) fn borrowed_logical_sparse_products() {
    use runmat_value::SparseTensor;
    let logical = LogicalArray::new(vec![1, 1, 0, 1], vec![2, 2]).unwrap();
    let pattern = SparseTensor::new_logical(2, 2, vec![0, 2, 3], vec![0, 1, 1]).unwrap();
    let full = Tensor::new(vec![2., 4., 3., 5.], vec![2, 2]).unwrap();
    let numeric =
        SparseTensor::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![2., 4., 3., 5.]).unwrap();
    for (a, b, expected) in [
        (
            Value::SparseTensor(pattern.clone()),
            Value::Tensor(full.clone()),
            vec![2., 6., 3., 8.],
        ),
        (
            Value::Tensor(full),
            Value::SparseTensor(pattern.clone()),
            vec![5., 9., 3., 5.],
        ),
        (
            Value::LogicalArray(logical.clone()),
            Value::SparseTensor(numeric.clone()),
            vec![2., 6., 3., 8.],
        ),
        (
            Value::SparseTensor(numeric),
            Value::LogicalArray(logical),
            vec![5., 9., 3., 5.],
        ),
    ] {
        let Value::Tensor(result) = mtimes_builtin(a, b).unwrap() else {
            panic!("full borrowed-logical product");
        };
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(result.materialize_f64(), expected);
    }
    let Value::SparseTensor(result) = mtimes_builtin(
        Value::SparseTensor(pattern.clone()),
        Value::SparseTensor(pattern),
    )
    .unwrap() else {
        panic!("numeric CSC from logical pattern product");
    };
    assert!(!result.is_logical());
    assert_eq!(result.as_f64_slice(), Some(&[1., 2., 1.][..]));
    assert_eq!(&result.col_ptrs[..], &[0, 2, 3]);
    assert_eq!(&result.row_indices[..], &[0, 1, 1]);
}
