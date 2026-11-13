use super::*;

/// Character geometry and raw UTF-16 codes enter the same dense/CSC product.
pub(super) fn character_numeric_products() {
    use runmat_value::{CharArray, SparseTensor, Utf16CodeUnit};
    let character = |codes: &[u16], rows, cols| {
        Value::CharArray(
            CharArray::from_code_units(
                codes.iter().copied().map(Utf16CodeUnit::from_raw).collect(),
                rows,
                cols,
            )
            .unwrap(),
        )
    };
    let chars = character(&[65, 233, 0xD800, 4], 2, 2);
    let dense = Value::Tensor(Tensor::new(vec![1., 3., 2., 4.], vec![2, 2]).unwrap());
    let sparse = Value::SparseTensor(
        SparseTensor::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1., 3., 2., 4.]).unwrap(),
    );
    for numeric in [dense.clone(), sparse.clone()] {
        for (left, right, expected) in [
            (
                chars.clone(),
                numeric.clone(),
                vec![764., 55308., 1062., 110608.],
            ),
            (
                numeric.clone(),
                chars.clone(),
                vec![110657., 221379., 241., 715.],
            ),
        ] {
            let Value::Tensor(result) = mtimes_builtin(left, right).unwrap() else {
                panic!("character matrix product must be double");
            };
            assert_eq!(result.shape(), [2, 2]);
            assert_eq!(result.materialize_f64(), expected);
        }
        for (left, right) in [
            (character(&[233], 1, 1), numeric.clone()),
            (numeric, character(&[233], 1, 1)),
        ] {
            let result = mtimes_builtin(left, right).unwrap();
            let values = match result {
                Value::Tensor(value) => value.materialize_f64(),
                Value::SparseTensor(value) => value.materialize_f64(),
                _ => panic!("character scalar product must be numeric"),
            };
            assert_eq!(values, vec![233., 699., 466., 932.]);
        }
    }
    let result = mtimes_builtin(character(&[233], 1, 1), character(&[65], 1, 1)).unwrap();
    assert!(matches!(result, Value::Num(value) if value == 15145.));
}
