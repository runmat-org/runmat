use super::*;
use runmat_value::{ComplexTensor, SparseTensor, Tensor};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dense_and_csc_triangular_substitution_preserve_class_and_orientation() {
    let dense = Value::Tensor(Tensor::new(vec![2.0, 3.0, 0.0, 4.0], vec![2, 2]).unwrap());
    let sparse = Value::SparseTensor(
        SparseTensor::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0, 3.0, 4.0]).unwrap(),
    );
    for matrix in [&dense, &sparse] {
        let TriangularLinearOperator::RealF64(lower) =
            TriangularLinearOperator::borrow_unbounded(matrix, Triangle::Lower, Diagonal::Explicit)
                .unwrap()
        else {
            panic!("double lower class changed");
        };
        let mut normal = [0.0; 2];
        super::product::apply_into_unbounded(&lower, &[2.0, 11.0], &mut normal).unwrap();
        assert_eq!(normal, [1.0, 2.0]);
        let mut transposed = [0.0; 2];
        super::product::apply_transpose_into_unbounded(&lower, &[8.0, 8.0], &mut transposed)
            .unwrap();
        assert_eq!(transposed, [1.0, 2.0]);
    }

    let omitted_unit = Value::SparseTensor(
        SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![1], vec![2.0]).unwrap(),
    );
    let TriangularLinearOperator::RealF32(unit) =
        TriangularLinearOperator::borrow_unbounded(&omitted_unit, Triangle::Lower, Diagonal::Unit)
            .unwrap()
    else {
        panic!("single unit lower class changed");
    };
    let mut unit_result = [0.0_f32; 2];
    super::product::apply_into_unbounded(&unit, &[3.0, 10.0], &mut unit_result).unwrap();
    assert_eq!(unit_result, [3.0, 4.0]);
    assert!(matches!(
        TriangularLinearOperator::borrow_unbounded(
            &omitted_unit,
            Triangle::Lower,
            Diagonal::Explicit
        ),
        Err(TriangularOperatorError::MissingDiagonal(0))
    ));

    let complex = Value::ComplexTensor(
        ComplexTensor::from_f32(
            vec![(2.0, 0.0), (0.0, 1.0), (0.0, 0.0), (3.0, 0.0)],
            vec![2, 2],
        )
        .unwrap(),
    );
    let TriangularLinearOperator::ComplexF32(lower) =
        TriangularLinearOperator::borrow_unbounded(&complex, Triangle::Lower, Diagonal::Explicit)
            .unwrap()
    else {
        panic!("complex single class changed");
    };
    let mut adjoint = [ComplexElement(0.0_f32, 0.0); 2];
    super::product::apply_adjoint_into_unbounded(
        &lower,
        &[ComplexElement(2.0, 0.0), ComplexElement(6.0, 0.0)],
        &mut adjoint,
    )
    .unwrap();
    assert_eq!(
        adjoint,
        [ComplexElement(1.0, 1.0), ComplexElement(2.0, 0.0)]
    );

    let extreme = Value::ComplexTensor(ComplexTensor::new(vec![(0.75, 0.75)], vec![1, 1]).unwrap());
    let TriangularLinearOperator::ComplexF64(extreme) =
        TriangularLinearOperator::borrow_unbounded(&extreme, Triangle::Lower, Diagonal::Explicit)
            .unwrap()
    else {
        panic!("complex double class changed");
    };
    let mut finite_quotient = [ComplexElement(0.0, 0.0)];
    super::product::apply_into_unbounded(
        &extreme,
        &[ComplexElement(1.7e308, 0.0)],
        &mut finite_quotient,
    )
    .unwrap();
    let expected = 1.7e308 / 1.5;
    assert!((finite_quotient[0].0 / expected - 1.0).abs() < 4.0 * f64::EPSILON);
    assert!((finite_quotient[0].1 / -expected - 1.0).abs() < 4.0 * f64::EPSILON);

    let huge_diagonal =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0e308, 0.0)], vec![1, 1]).unwrap());
    let TriangularLinearOperator::ComplexF64(huge_diagonal) =
        TriangularLinearOperator::borrow_unbounded(
            &huge_diagonal,
            Triangle::Lower,
            Diagonal::Explicit,
        )
        .unwrap()
    else {
        panic!("complex double class changed");
    };
    let mixed_scale = ComplexElement(1.0e308, 1.0);
    let mut retained_component = [ComplexElement(0.0, 0.0)];
    super::product::apply_into_unbounded(&huge_diagonal, &[mixed_scale], &mut retained_component)
        .unwrap();
    assert!((retained_component[0].0 - 1.0).abs() < 4.0 * f64::EPSILON);
    assert!(retained_component[0].1 != 0.0);
    assert!((retained_component[0].1 / 1.0e-308 - 1.0).abs() < 4.0 * f64::EPSILON);

    let identity = Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap());
    let TriangularLinearOperator::ComplexF64(identity) =
        TriangularLinearOperator::borrow_unbounded(&identity, Triangle::Lower, Diagonal::Explicit)
            .unwrap()
    else {
        panic!("complex double class changed");
    };
    let mixed_numerator = ComplexElement(1.0e308, 1.0e-308);
    let mut retained_numerator = [ComplexElement(0.0, 0.0)];
    super::product::apply_into_unbounded(&identity, &[mixed_numerator], &mut retained_numerator)
        .unwrap();
    assert_eq!(retained_numerator, [mixed_numerator]);
}
