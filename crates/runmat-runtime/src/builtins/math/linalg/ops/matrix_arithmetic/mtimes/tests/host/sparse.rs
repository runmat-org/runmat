use super::*;

/// Same public multiplication boundary, four native storage classes and both
/// mixed orientations; CSC/CSC checks stored output without a dense conversion.
pub(super) fn sparse_storage_products() {
    use runmat_value::{ComplexTensor, NumericDType, SparseTensor};
    let scale =
        Value::SparseTensor(SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![2.]).unwrap());
    for (a, b) in [
        (
            scale.clone(),
            Value::Tensor(Tensor::new(vec![1., 2.], vec![2, 1]).unwrap()),
        ),
        (
            Value::Tensor(Tensor::new(vec![1., 2.], vec![2, 1]).unwrap()),
            scale,
        ),
    ] {
        let value = mtimes_builtin(a, b).unwrap();
        match value {
            Value::Tensor(value) => assert_eq!(value.materialize_f64(), vec![2., 4.]),
            Value::SparseTensor(value) => {
                assert_eq!(value.get(0, 0), Some(2.));
                assert_eq!(value.get(1, 0), Some(4.));
            }
            other => panic!("scaled floating matrix: {other:?}"),
        }
    }
    for single in [false, true] {
        let dtype = if single {
            NumericDType::F32
        } else {
            NumericDType::F64
        };
        // Noncommuting matrices: AB cancels (0,0), merges several stored
        // products, and discovers rows in nonsorted order before CSC emission.
        let av = vec![1., 4., 2., 1., 3., 1.];
        let bv = vec![2., -1., 1., 3., 2., 1.];
        let sparse = if single {
            SparseTensor::new_f32(
                3,
                3,
                vec![0, 2, 4, 6],
                vec![0, 2, 0, 1, 1, 2],
                av.iter().map(|v| *v as f32).collect(),
            )
            .unwrap()
        } else {
            SparseTensor::new(3, 3, vec![0, 2, 4, 6], vec![0, 2, 0, 1, 1, 2], av).unwrap()
        };
        let other = if single {
            SparseTensor::new_f32(
                3,
                3,
                vec![0, 2, 4, 6],
                vec![0, 1, 0, 2, 1, 2],
                bv.iter().map(|v| *v as f32).collect(),
            )
            .unwrap()
        } else {
            SparseTensor::new(3, 3, vec![0, 2, 4, 6], vec![0, 1, 0, 2, 1, 2], bv).unwrap()
        };
        let bd = vec![2., -1., 0., 1., 0., 3., 0., 2., 1.];
        let dense = if single {
            Tensor::from_f32(bd.iter().map(|v| *v as f32).collect(), vec![3, 3]).unwrap()
        } else {
            Tensor::new(bd, vec![3, 3]).unwrap()
        };
        for (a, b, expected) in [
            (
                Value::SparseTensor(sparse.clone()),
                Value::Tensor(dense.clone()),
                vec![0., -1., 8., 1., 9., 7., 4., 5., 1.],
            ),
            (
                Value::Tensor(dense),
                Value::SparseTensor(sparse.clone()),
                vec![2., 7., 4., 5., -2., 3., 3., 2., 10.],
            ),
        ] {
            let Value::Tensor(result) = mtimes_builtin(a, b).unwrap() else {
                panic!("full product");
            };
            assert_eq!(result.numeric_dtype(), dtype);
            assert_eq!(result.materialize_f64(), expected);
        }
        let Value::SparseTensor(result) =
            mtimes_builtin(Value::SparseTensor(sparse), Value::SparseTensor(other)).unwrap()
        else {
            panic!("CSC product");
        };
        assert_eq!(result.numeric_dtype(), Some(dtype));
        assert_eq!(&result.col_ptrs[..], &[0, 2, 5, 8]);
        assert_eq!(&result.row_indices[..], &[1, 2, 0, 1, 2, 0, 1, 2]);
        let values = if single {
            result
                .as_f32_slice()
                .unwrap()
                .iter()
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>()
        } else {
            result.as_f64_slice().unwrap().to_vec()
        };
        assert_eq!(values, vec![-1., 8., 1., 9., 7., 4., 5., 1.]);
        let sparse = if single {
            SparseTensor::new_complex_f32(
                2,
                2,
                vec![0, 1, 2],
                vec![0, 1],
                vec![(2., 1.), (3., -2.)],
            )
            .unwrap()
        } else {
            SparseTensor::new_complex(2, 2, vec![0, 1, 2], vec![0, 1], vec![(2., 1.), (3., -2.)])
                .unwrap()
        };
        let dense = ComplexTensor::from_f64_values_with_dtype(
            vec![(2., 1.), (0., 0.), (0., 0.), (3., -2.)],
            vec![2, 2],
            dtype,
        )
        .unwrap();
        for (a, b) in [
            (
                Value::SparseTensor(sparse.clone()),
                Value::ComplexTensor(dense.clone()),
            ),
            (
                Value::ComplexTensor(dense),
                Value::SparseTensor(sparse.clone()),
            ),
        ] {
            let Value::ComplexTensor(result) = mtimes_builtin(a, b).unwrap() else {
                panic!("complex full product");
            };
            assert_eq!(result.numeric_dtype(), dtype);
            assert_eq!(
                result.materialize_f64(),
                vec![(3., 4.), (0., 0.), (0., 0.), (5., -12.)]
            );
        }
        let Value::SparseTensor(result) = mtimes_builtin(
            Value::SparseTensor(sparse.clone()),
            Value::SparseTensor(sparse),
        )
        .unwrap() else {
            panic!("complex CSC product");
        };
        let values = if single {
            result
                .as_complex_f32_slice()
                .unwrap()
                .iter()
                .map(|v| (f64::from(v.0), f64::from(v.1)))
                .collect::<Vec<_>>()
        } else {
            result
                .as_complex_f64_slice()
                .unwrap()
                .iter()
                .map(|v| (v.0, v.1))
                .collect()
        };
        assert_eq!(values, vec![(3., 4.), (5., -12.)]);
    }
    let Value::SparseTensor(empty) = mtimes_builtin(
        Value::SparseTensor(SparseTensor::new(2, 0, vec![0], vec![], vec![]).unwrap()),
        Value::SparseTensor(SparseTensor::new(0, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap()),
    )
    .unwrap() else {
        panic!("empty CSC product");
    };
    assert_eq!((empty.rows, empty.cols, empty.nnz()), (2, 3, 0));
}
