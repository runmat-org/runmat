use super::*;
use runmat_value::{ComplexTensor, SparseTensor, Tensor};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dense_and_csc_share_native_product_and_orientation_contract() {
    // Native f32 column-major columns [1,2], [3,4], [5,6].
    let dense =
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap());
    let sparse = Value::SparseTensor(
        SparseTensor::new_f32(
            2,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap(),
    );
    for coefficient in [&dense, &sparse] {
        let LinearOperator::RealF32(operator) =
            LinearOperator::borrow_unbounded(coefficient).unwrap()
        else {
            panic!("matrix class was converted");
        };
        assert_eq!((operator.rows(), operator.cols()), (2, 3));
        let mut normal = [0.0; 2];
        operator
            .apply_into_unbounded(&[10.0, 20.0, 30.0], &mut normal)
            .unwrap();
        assert_eq!(normal, [220.0, 280.0]);
        let mut transposed = [0.0; 3];
        operator
            .apply_transpose_into_unbounded(&[7.0, 11.0], &mut transposed)
            .unwrap();
        assert_eq!(transposed, [29.0, 65.0, 101.0]);
    }

    let complex = Value::ComplexTensor(
        ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![2, 1]).unwrap(),
    );
    let LinearOperator::ComplexF64(operator) = LinearOperator::borrow_unbounded(&complex).unwrap()
    else {
        panic!("complex matrix class was converted");
    };
    let mut normal = [ComplexElement(0.0, 0.0); 2];
    operator
        .apply_into_unbounded(&[ComplexElement(3.0, 2.0)], &mut normal)
        .unwrap();
    assert_eq!(normal, [ComplexElement(1.0, 5.0), ComplexElement(8.0, 1.0)]);
    let mut transposed = [ComplexElement(0.0, 0.0)];
    let mut adjoint = [ComplexElement(0.0, 0.0)];
    let input = [ComplexElement(1.0, 1.0), ComplexElement(2.0, 0.0)];
    operator
        .apply_transpose_into_unbounded(&input, &mut transposed)
        .unwrap();
    operator
        .apply_adjoint_into_unbounded(&input, &mut adjoint)
        .unwrap();
    assert_eq!(transposed, [ComplexElement(4.0, 0.0)]);
    assert_eq!(adjoint, [ComplexElement(6.0, 2.0)]);

    let rank_three = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap());
    assert!(matches!(
        LinearOperator::borrow_unbounded(&rank_three),
        Err(LinearOperatorError::DimensionMismatch)
    ));
}
