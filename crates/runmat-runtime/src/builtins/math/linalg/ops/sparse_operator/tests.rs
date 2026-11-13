use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn csc_product_preserves_native_real_and_complex_precision() {
    let f32_matrix =
        SparseTensor::new_f32(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![2.0, 3.0, 4.0]).unwrap();
    let SparseLinearOperator::RealF32(f32_op) =
        SparseLinearOperator::borrow_unbounded(&f32_matrix).unwrap()
    else {
        panic!("single CSC class changed");
    };
    let mut f32_result = [0.0; 3];
    f32_op
        .apply_into_unbounded(&[5.0, 7.0], &mut f32_result)
        .unwrap();
    assert_eq!(f32_result, [10.0, 28.0, 15.0]);
    assert!(f32_op.rows() == 3 && f32_op.cols() == 2 && f32_op.nnz() == 3);
    assert!(matches!(
        SparseLinearOperator::borrow_unbounded(&f32_matrix)
            .unwrap()
            .scratch_plan(usize::MAX),
        Err(SparseOperatorError::Overflow)
    ));

    let f64_matrix =
        SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![2.0, 3.0, 4.0]).unwrap();
    let SparseLinearOperator::RealF64(f64_op) =
        SparseLinearOperator::borrow_unbounded(&f64_matrix).unwrap()
    else {
        panic!("double CSC class changed");
    };
    let mut f64_result = [0.0; 3];
    f64_op
        .apply_into_unbounded(&[5.0, 7.0], &mut f64_result)
        .unwrap();
    assert_eq!(f64_result, [10.0, 28.0, 15.0]);

    let complex32 = SparseTensor::new_complex_f32(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 1.0), (0.0, 3.0)],
    )
    .unwrap();
    let SparseLinearOperator::ComplexF32(complex32_op) =
        SparseLinearOperator::borrow_unbounded(&complex32).unwrap()
    else {
        panic!("complex single CSC class changed");
    };
    let mut complex32_result = [ComplexElement(0.0, 0.0); 2];
    complex32_op
        .apply_into_unbounded(
            &[ComplexElement(1.0, 2.0), ComplexElement(2.0, -1.0)],
            &mut complex32_result,
        )
        .unwrap();
    assert_eq!(
        complex32_result,
        [ComplexElement(0.0, 5.0), ComplexElement(3.0, 6.0)]
    );

    let complex64 = SparseTensor::new_complex(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 1.0), (0.0, 3.0)],
    )
    .unwrap();
    let SparseLinearOperator::ComplexF64(complex64_op) =
        SparseLinearOperator::borrow_unbounded(&complex64).unwrap()
    else {
        panic!("complex double CSC class changed");
    };
    let mut complex64_result = [ComplexElement(0.0, 0.0); 2];
    complex64_op
        .apply_into_unbounded(
            &[ComplexElement(1.0, 2.0), ComplexElement(2.0, -1.0)],
            &mut complex64_result,
        )
        .unwrap();
    assert_eq!(
        complex64_result,
        [ComplexElement(0.0, 5.0), ComplexElement(3.0, 6.0)]
    );
    let mut transposed = [ComplexElement(0.0, 0.0); 2];
    let mut adjointed = [ComplexElement(0.0, 0.0); 2];
    complex64_op
        .apply_transpose_into_unbounded(
            &[ComplexElement(1.0, 0.0), ComplexElement(0.0, 0.0)],
            &mut transposed,
        )
        .unwrap();
    complex64_op
        .apply_adjoint_into_unbounded(
            &[ComplexElement(1.0, 0.0), ComplexElement(0.0, 0.0)],
            &mut adjointed,
        )
        .unwrap();
    assert_eq!(
        transposed,
        [ComplexElement(2.0, 1.0), ComplexElement(0.0, 0.0)]
    );
    assert_eq!(
        adjointed,
        [ComplexElement(2.0, -1.0), ComplexElement(0.0, 0.0)]
    );
    assert!(matches!(
        complex64_op.apply_into_unbounded(&[], &mut complex64_result),
        Err(SparseOperatorError::DimensionMismatch)
    ));

    let mut invalid = f64_matrix;
    invalid.col_ptrs[1] = 4;
    assert!(matches!(
        SparseLinearOperator::borrow_unbounded(&invalid),
        Err(SparseOperatorError::InvalidTopology(_))
    ));
}
