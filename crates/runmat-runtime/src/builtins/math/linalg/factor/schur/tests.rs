use super::*;

#[test]
fn real_schur_retains_two_by_two_blocks_and_reconstructs() {
    let source = DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0]);
    let SchurForm::Real {
        vectors,
        triangular,
    } = SchurForm::real(source.clone()).unwrap()
    else {
        panic!("real source must retain real Schur form");
    };
    assert!((&vectors * &triangular * vectors.transpose() - source).norm() < 1e-10);
    assert!((vectors.transpose() * &vectors - DMatrix::<f64>::identity(3, 3)).norm() < 1e-10);
    let eigenvalues = form::real_eigenvalues(&triangular);
    assert!(eigenvalues
        .iter()
        .any(|value| (value.im.abs() - 1.0).abs() < 1e-10));
}

#[test]
fn tiny_real_schur_subdiagonal_is_structural_not_discarded() {
    let matrix = DMatrix::from_row_slice(2, 2, &[0.0, 1e20, -1e-20, 0.0]);
    assert_eq!(form::real_blocks(&matrix)[0].size, 2);
    let values = form::real_eigenvalues(&matrix);
    assert!((values[0].im.abs() - 1.0).abs() < 1e-12);
    assert!((values[1].im.abs() - 1.0).abs() < 1e-12);
}

#[test]
fn complexification_preserves_factorization() {
    let source = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
    let form = SchurForm::real(source.clone())
        .unwrap()
        .into_complex()
        .unwrap();
    let SchurForm::Complex {
        vectors,
        triangular,
    } = form
    else {
        panic!("complexification must produce complex factors");
    };
    let source = source.map(|value| Complex64::new(value, 0.0));
    assert!((&vectors * &triangular * vectors.adjoint() - source).norm() < 1e-10);
    assert!(triangular[(1, 0)].norm() < 1e-12);
}

#[test]
fn adjacent_exchange_preserves_real_schur_reconstruction_and_order() {
    let source = DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 1.0, 0.0, -3.0, 2.0, 0.0, 0.0, 1.0]);
    let mut form = SchurForm::real(source.clone()).unwrap();
    form.reorder(&[2, 1, 0]).unwrap();
    assert!((form.eigenvalues()[0].re - 1.0).abs() < 1e-10);
    let SchurForm::Real {
        vectors,
        triangular,
    } = form
    else {
        panic!("real reordering must retain real factors");
    };
    assert!((&vectors * &triangular * vectors.transpose() - source).norm() < 1e-9);
    assert!((vectors.transpose() * &vectors - DMatrix::<f64>::identity(3, 3)).norm() < 1e-10);
}

#[test]
fn real_pair_moves_as_one_block() {
    let triangular = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 2.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
    let source = triangular.clone();
    let mut form = SchurForm::from_real_parts(DMatrix::identity(3, 3), triangular).unwrap();
    assert_eq!(
        form.blocks()
            .iter()
            .map(|block| block.size)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    form.reorder(&[1, 0]).unwrap();
    let eigenvalues = form.eigenvalues();
    assert!((eigenvalues[0].im.abs() - 1.0).abs() < 1e-10);
    assert!((eigenvalues[1].im.abs() - 1.0).abs() < 1e-10);
    let SchurForm::Real {
        vectors,
        triangular,
    } = form
    else {
        panic!("real pair must remain real");
    };
    assert!((&vectors * &triangular * vectors.transpose() - source).norm() < 1e-9);
}

#[test]
fn complex_singleton_exchange_preserves_unitarity() {
    let triangular = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(3.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-2.0, 2.0),
        ],
    );
    let mut form =
        SchurForm::from_complex_parts(DMatrix::identity(2, 2), triangular.clone()).unwrap();
    form.reorder(&[1, 0]).unwrap();
    let SchurForm::Complex {
        vectors,
        triangular: reordered,
    } = form
    else {
        panic!("complex form must remain complex");
    };
    assert!((reordered[(0, 0)] - Complex64::new(-2.0, 2.0)).norm() < 1e-10);
    assert!((&vectors * &reordered * vectors.adjoint() - triangular).norm() < 1e-10);
    assert!((vectors.adjoint() * &vectors - DMatrix::<Complex64>::identity(2, 2)).norm() < 1e-10);
}

#[test]
fn identical_separated_blocks_and_large_finite_coefficients_can_exchange() {
    let identical = DMatrix::from_diagonal_element(2, 2, 3.0);
    let mut form = SchurForm::from_real_parts(DMatrix::identity(2, 2), identical.clone()).unwrap();
    form.reorder(&[1, 0]).unwrap();
    let SchurForm::Real {
        vectors,
        triangular,
    } = form
    else {
        panic!("real form expected")
    };
    assert!((&vectors * &triangular * vectors.transpose() - identical).norm() < 1e-12);

    let large = DMatrix::from_row_slice(2, 2, &[1e300, 2e300, 0.0, -1e300]);
    let mut form = SchurForm::from_real_parts(DMatrix::identity(2, 2), large.clone()).unwrap();
    form.reorder(&[1, 0]).unwrap();
    let SchurForm::Real {
        vectors,
        triangular,
    } = form
    else {
        panic!("real form expected")
    };
    let residual = (&vectors * &triangular * vectors.transpose() - large)
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        / 2e300;
    assert!(
        residual < 1e-10,
        "scaled reconstruction residual: {residual}"
    );
}
