use faer::{c64, Col, Mat};

use super::algorithm::{complex_qz, real_qz};
use super::eigvals::{complex_schur_values, real_schur_values};
use super::eigvectors::complex_eigenvectors;
use super::real_eigenvectors::real_eigenvectors;
use super::real_reorder::reorder_real_clusters;
use super::reorder::{reorder_clusters, ClusterOrder};

#[test]
fn qz_resource_geometry_rejects_overflow_before_matrix_construction() {
    use runmat_value::NumericDType;

    assert!(
        super::allocation::factor_plan(usize::MAX, NumericDType::F64, super::Mode::Real, 6)
            .is_err()
    );
    assert!(super::allocation::reorder_plan(
        usize::MAX,
        NumericDType::F64,
        super::Mode::Complex,
        4
    )
    .is_err());
    assert!(super::allocation::eigenvalues_plan(
        usize::MAX,
        NumericDType::F32,
        super::Mode::Real,
        true
    )
    .is_err());
}

fn close(a: c64, b: c64, tolerance: f64) {
    assert!(
        (a.re - b.re).hypot(a.im - b.im) <= tolerance,
        "{a:?} != {b:?}"
    );
}

fn check_pencil_reconstruction(
    original_a: &Mat<c64>,
    original_b: &Mat<c64>,
    form: &super::algorithm::ComplexQz,
) {
    let n = original_a.nrows();
    for row in 0..n {
        for col in 0..n {
            let reconstruct = |schur: &Mat<c64>| {
                let mut value = c64::new(0.0, 0.0);
                for i in 0..n {
                    for j in 0..n {
                        let z = form.right[(col, j)];
                        value += form.left[(row, i)] * schur[(i, j)] * c64::new(z.re, -z.im);
                    }
                }
                value
            };
            close(reconstruct(&form.aa), original_a[(row, col)], 1e-8);
            close(reconstruct(&form.bb), original_b[(row, col)], 1e-8);
        }
    }
}

#[test]
fn complex_qz_reconstructs_singular_b_without_inverse() {
    let a = Mat::from_fn(2, 2, |i, j| {
        let values = [[2.0, 0.0], [1.0, 3.0]];
        c64::new(values[j][i], 0.0)
    });
    let b = Mat::from_fn(2, 2, |i, j| {
        let values = [[1.0, 0.0], [0.0, 0.0]];
        c64::new(values[j][i], 0.0)
    });
    let form = complex_qz(a.clone(), b.clone()).expect("singular B is a valid QZ pencil");
    check_pencil_reconstruction(&a, &b, &form);
}

#[test]
fn complex_qz_reorder_moves_selected_infinite_eigenvalue_with_both_unitaries() {
    let a = Mat::from_fn(2, 2, |i, j| {
        let values = [[2.0, 0.0], [0.0, 3.0]];
        c64::new(values[j][i], 0.0)
    });
    let b = Mat::from_fn(2, 2, |i, j| {
        let values = [[1.0, 0.0], [0.0, 0.0]];
        c64::new(values[j][i], 0.0)
    });
    let mut form = complex_qz(a.clone(), b.clone()).expect("QZ");
    let selected = (0..2)
        .map(|i| {
            if form.bb[(i, i)].re.hypot(form.bb[(i, i)].im) == 0.0 {
                0
            } else {
                1
            }
        })
        .collect::<Vec<_>>();
    reorder_clusters(&mut form, &selected, ClusterOrder::SelectedFirst)
        .expect("separated diagonal blocks can exchange");
    assert_eq!(form.bb[(0, 0)].re.hypot(form.bb[(0, 0)].im), 0.0);
    check_pencil_reconstruction(&a, &b, &form);
}

#[test]
fn ordeig_retains_real_conjugate_block_order_and_infinity() {
    let a = Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
        (0, 1) => -1.0,
        (1, 0) => 1.0,
        (2, 2) => 4.0,
        _ => 0.0,
    });
    let b = Mat::<f64>::from_fn(3, 3, |i, j| if i == j && i != 2 { 1.0 } else { 0.0 });
    let values = real_schur_values(&a, &b).expect("quasi-triangular pencil");
    assert_eq!(values.len(), 3);
    assert!(values[0].0.abs() < 1e-10 && values[1].0.abs() < 1e-10);
    assert!((values[0].1.abs() - 1.0).abs() < 1e-10);
    assert!((values[1].1.abs() - 1.0).abs() < 1e-10);
    assert!(values[0].1 * values[1].1 < 0.0);
    assert!(values[2].0.is_infinite());

    let complex_a = Mat::from_fn(1, 1, |_, _| c64::new(0.0, 0.0));
    let complex_b = Mat::from_fn(1, 1, |_, _| c64::new(0.0, 0.0));
    let indeterminate = complex_schur_values(&complex_a, &complex_b).expect("singular pencil");
    assert!(indeterminate[0].0.is_nan());

    let huge = Mat::from_fn(1, 1, |_, _| c64::new(1e308, 1e308));
    let finite_ratio = complex_schur_values(&huge, &huge).expect("scaled ratio");
    assert!((finite_ratio[0].0 - 1.0).abs() < 1e-12);
    assert!(finite_ratio[0].1.abs() < 1e-12);
}

#[test]
fn retained_qz_factors_yield_ordered_left_and_right_eigenvectors() {
    let a = Mat::from_fn(2, 2, |i, j| {
        let entries = [[2.0, 1.0], [0.0, 3.0]];
        c64::new(entries[i][j], 0.0)
    });
    let b = Mat::from_fn(2, 2, |i, j| {
        if i == j {
            c64::new(1.0, 0.0)
        } else {
            c64::new(0.0, 0.0)
        }
    });
    let form = complex_qz(a.clone(), b.clone()).expect("QZ");
    let vectors = complex_eigenvectors(&form, true).expect("retained-factor solves");
    let left = vectors.left.expect("left requested");
    for k in 0..2 {
        let alpha = form.alpha[k];
        let beta = form.beta[k];
        for row in 0..2 {
            let mut right_residual = c64::new(0.0, 0.0);
            let mut left_residual = c64::new(0.0, 0.0);
            for col in 0..2 {
                right_residual +=
                    (beta * a[(row, col)] - alpha * b[(row, col)]) * vectors.right[(col, k)];
                let l = left[(col, k)];
                left_residual +=
                    c64::new(l.re, -l.im) * (beta * a[(col, row)] - alpha * b[(col, row)]);
            }
            assert!(right_residual.re.hypot(right_residual.im) < 1e-8);
            assert!(left_residual.re.hypot(left_residual.im) < 1e-8);
        }
    }
}

#[test]
fn retained_real_qz_pair_encodes_conjugate_eigenvectors_without_refactorization() {
    let a = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
        (0, 1) => -1.0,
        (1, 0) => 1.0,
        _ => 0.0,
    });
    let b = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let form = real_qz(a.clone(), b.clone()).expect("real QZ");
    let vectors = real_eigenvectors(&form, true).expect("retained real block solves");
    let left = vectors.left.expect("left requested");
    for selected in 0..2 {
        let pair_sign = if selected == 0 { 1.0 } else { -1.0 };
        let alpha = c64::new(form.alpha_re[selected], form.alpha_im[selected]);
        let beta = c64::new(form.beta[selected], 0.0);
        for row in 0..2 {
            let mut right_residual = c64::new(0.0, 0.0);
            let mut left_residual = c64::new(0.0, 0.0);
            for col in 0..2 {
                let v = c64::new(vectors.right[(col, 0)], pair_sign * vectors.right[(col, 1)]);
                let w = c64::new(left[(col, 0)], pair_sign * left[(col, 1)]);
                right_residual += (beta * c64::new(a[(row, col)], 0.0)
                    - alpha * c64::new(b[(row, col)], 0.0))
                    * v;
                left_residual += c64::new(w.re, -w.im)
                    * (beta * c64::new(a[(col, row)], 0.0) - alpha * c64::new(b[(col, row)], 0.0));
            }
            assert!(right_residual.re.hypot(right_residual.im) < 1e-8);
            assert!(left_residual.re.hypot(left_residual.im) < 1e-8);
        }
    }
}

#[test]
fn defective_pencil_can_publish_dependent_but_residual_valid_vectors() {
    let form = super::algorithm::ComplexQz {
        aa: Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => c64::new(2.0, 0.0),
            (0, 1) => c64::new(1.0, 0.0),
            _ => c64::new(0.0, 0.0),
        }),
        bb: Mat::from_fn(2, 2, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0)),
        left: Mat::from_fn(2, 2, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0)),
        right: Mat::from_fn(2, 2, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0)),
        alpha: Col::from_fn(2, |_| c64::new(2.0, 0.0)),
        beta: Col::from_fn(2, |_| c64::new(1.0, 0.0)),
    };
    let vectors = complex_eigenvectors(&form, true).expect("defective basis is permitted");
    let left = vectors.left.expect("left requested");
    for col in 0..2 {
        assert!(vectors.right[(1, col)].re.hypot(vectors.right[(1, col)].im) < 1e-12);
        assert!(left[(0, col)].re.hypot(left[(0, col)].im) < 1e-12);
    }
}

#[test]
fn retained_eigenvector_residual_uses_the_pencil_scale() {
    let tiny = 1e-300;
    let form = super::algorithm::ComplexQz {
        aa: Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (0, 1) => c64::new(tiny, 0.0),
            (1, 1) => c64::new(2.0 * tiny, 0.0),
            _ => c64::new(0.0, 0.0),
        }),
        bb: Mat::from_fn(2, 2, |i, j| c64::new(if i == j { tiny } else { 0.0 }, 0.0)),
        left: Mat::from_fn(2, 2, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0)),
        right: Mat::from_fn(2, 2, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0)),
        alpha: Col::from_fn(2, |i| c64::new(if i == 0 { tiny } else { 2.0 * tiny }, 0.0)),
        beta: Col::from_fn(2, |_| c64::new(tiny, 0.0)),
    };
    let vectors = complex_eigenvectors(&form, true).expect("scaled retained-factor solve");
    assert!((vectors.right[(0, 1)].re - vectors.right[(1, 1)].re).abs() < 1e-12);
}

#[test]
fn tiny_nonzero_schur_coefficients_keep_their_eigenvector_direction() {
    let tiny = 1e-20;
    let identity_complex = Mat::from_fn(2, 2, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0));
    let complex = super::algorithm::ComplexQz {
        aa: Mat::from_fn(2, 2, |i, j| {
            c64::new(
                if (i, j) == (0, 1) || (i, j) == (1, 1) {
                    tiny
                } else {
                    0.0
                },
                0.0,
            )
        }),
        bb: identity_complex.clone(),
        left: identity_complex.clone(),
        right: identity_complex,
        alpha: Col::from_fn(2, |i| c64::new(if i == 1 { tiny } else { 0.0 }, 0.0)),
        beta: Col::from_fn(2, |_| c64::new(1.0, 0.0)),
    };
    let vectors = complex_eigenvectors(&complex, true).expect("tiny triangular solve");
    assert!((vectors.right[(0, 1)].re - vectors.right[(1, 1)].re).abs() < 1e-12);
    assert!(vectors.right[(0, 1)].re.abs() > 0.5);

    let identity_real = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let real = super::algorithm::RealQz {
        aa: Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 1) => -tiny,
            (1, 0) => tiny,
            _ => 0.0,
        }),
        bb: identity_real.clone(),
        left: identity_real.clone(),
        right: identity_real,
        alpha_re: Col::from_fn(2, |_| 0.0),
        alpha_im: Col::from_fn(2, |i| if i == 0 { tiny } else { -tiny }),
        beta: Col::from_fn(2, |_| 1.0),
    };
    let vectors = real_eigenvectors(&real, true).expect("tiny conjugate block");
    assert!(vectors.right[(0, 0)].abs() > 0.5);
    assert!(vectors.right[(1, 1)].abs() > 0.5);
}

#[test]
fn rounded_homogeneous_values_admit_small_backward_error_but_not_wrong_direction() {
    let rounded = 1.0 + 4.0 * f64::EPSILON;
    let complex = super::algorithm::ComplexQz {
        aa: Mat::from_fn(1, 1, |_, _| c64::new(1.0, 1.0)),
        bb: Mat::from_fn(1, 1, |_, _| c64::new(1.0, 0.0)),
        left: Mat::from_fn(1, 1, |_, _| c64::new(1.0, 0.0)),
        right: Mat::from_fn(1, 1, |_, _| c64::new(1.0, 0.0)),
        alpha: Col::from_fn(1, |_| c64::new(rounded, 1.0)),
        beta: Col::from_fn(1, |_| c64::new(1.0, 0.0)),
    };
    let vectors = complex_eigenvectors(&complex, true).expect("rounded complex pencil");
    assert_eq!(vectors.right.ncols(), 1);
    assert_eq!(vectors.left.expect("left vector").ncols(), 1);

    let real = super::algorithm::RealQz {
        aa: Mat::from_fn(1, 1, |_, _| 1.0),
        bb: Mat::from_fn(1, 1, |_, _| 1.0),
        left: Mat::from_fn(1, 1, |_, _| 1.0),
        right: Mat::from_fn(1, 1, |_, _| 1.0),
        alpha_re: Col::from_fn(1, |_| rounded),
        alpha_im: Col::from_fn(1, |_| 0.0),
        beta: Col::from_fn(1, |_| 1.0),
    };
    let vectors = real_eigenvectors(&real, true).expect("rounded real pencil");
    assert_eq!(vectors.right.ncols(), 1);
    assert_eq!(vectors.left.expect("left vector").ncols(), 1);

    // The V5 tiny-coefficient direction witness remains stringent: a row
    // with only the wrong off-diagonal contribution has relative error one.
    assert!(!super::eigvectors::homogeneous_row_residual(
        [(c64::new(1e-20, 0.0), c64::new(0.0, 0.0))],
        1e-12,
    ));
}

#[test]
fn real_ordqz_moves_infinite_block_past_conjugate_pair_without_complexifying() {
    let a = Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
        (0, 1) => -1.0,
        (1, 0) => 1.0,
        (2, 2) => 3.0,
        _ => 0.0,
    });
    let b = Mat::<f64>::from_fn(3, 3, |i, j| if i == j && i != 2 { 1.0 } else { 0.0 });
    let mut form = real_qz(a.clone(), b.clone()).expect("real QZ");
    reorder_real_clusters(&mut form, &[1, 1, 0], ClusterOrder::SelectedFirst)
        .expect("separated real Schur blocks");
    assert!(form.bb[(0, 0)].abs() < 1e-8);
    for row in 0..3 {
        for col in 0..3 {
            let reconstruct = |schur: &Mat<f64>| {
                (0..3)
                    .flat_map(|i| (0..3).map(move |j| (i, j)))
                    .map(|(i, j)| form.left[(row, i)] * schur[(i, j)] * form.right[(col, j)])
                    .sum::<f64>()
            };
            assert!((reconstruct(&form.aa) - a[(row, col)]).abs() < 1e-8);
            assert!((reconstruct(&form.bb) - b[(row, col)]).abs() < 1e-8);
            if row > col + 1 {
                assert!(form.aa[(row, col)].abs() < 1e-8);
            }
            if row > col {
                assert!(form.bb[(row, col)].abs() < 1e-8);
            }
        }
    }
}

#[test]
fn real_ordqz_selector_keeps_conjugate_pair_in_one_cluster() {
    let a = Mat::<f64>::from_fn(3, 3, |i, j| match (i, j) {
        (0, 1) => -1.0,
        (1, 0) => 1.0,
        (2, 2) => 2.0,
        _ => 0.0,
    });
    let mut logical = vec![1, 0, 1];
    super::ordqz::pair_labels(&mut logical, &a, ClusterOrder::SelectedFirst);
    assert_eq!(logical, [0, 0, 1]);
    let mut clusters = vec![1, 3, 2];
    super::ordqz::pair_labels(&mut clusters, &a, ClusterOrder::NumericDescending);
    assert_eq!(clusters, [3, 3, 2]);
}

#[test]
fn numeric_ordqz_clusters_descend_in_both_factor_domains() {
    let real_a = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
    let real_b = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    let identity_real = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut real = super::algorithm::RealQz {
        aa: real_a,
        bb: real_b,
        left: identity_real.clone(),
        right: identity_real,
        alpha_re: Col::from_fn(3, |i| (i + 1) as f64),
        alpha_im: Col::zeros(3),
        beta: Col::from_fn(3, |_| 1.0),
    };
    reorder_real_clusters(&mut real, &[1, 3, 2], ClusterOrder::NumericDescending)
        .expect("real numeric clusters");
    assert!((real.aa[(0, 0)] / real.bb[(0, 0)] - 2.0).abs() < 1e-12);

    let complex_a = Mat::<c64>::from_fn(3, 3, |i, j| {
        c64::new(if i == j { (i + 1) as f64 } else { 0.0 }, 0.0)
    });
    let complex_b = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0));
    let identity_complex =
        Mat::<c64>::from_fn(3, 3, |i, j| c64::new(if i == j { 1.0 } else { 0.0 }, 0.0));
    let mut complex = super::algorithm::ComplexQz {
        aa: complex_a,
        bb: complex_b,
        left: identity_complex.clone(),
        right: identity_complex,
        alpha: Col::from_fn(3, |i| c64::new((i + 1) as f64, 0.0)),
        beta: Col::from_fn(3, |_| c64::new(1.0, 0.0)),
    };
    reorder_clusters(&mut complex, &[1, 3, 2], ClusterOrder::NumericDescending)
        .expect("complex numeric clusters");
    let ratio = super::eigvals::complex_schur_values(&complex.aa, &complex.bb)
        .expect("reordered complex Schur form");
    assert!((ratio[0].0 - 2.0).abs() < 1e-12);
}

#[test]
fn real_retained_vectors_accept_zero_rhs_free_component() {
    let form = super::algorithm::RealQz {
        aa: Mat::from_fn(2, 2, |i, j| {
            if i == j {
                if i == 0 {
                    1e300
                } else {
                    1.0
                }
            } else {
                0.0
            }
        }),
        bb: Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }),
        left: Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }),
        right: Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }),
        alpha_re: Col::from_fn(2, |i| if i == 0 { 1e300 } else { 1.0 }),
        alpha_im: Col::zeros(2),
        beta: Col::from_fn(2, |_| 1.0),
    };
    let vectors = real_eigenvectors(&form, true).expect("zero RHS has a free zero component");
    let left = vectors.left.expect("left vectors requested");
    assert!(left[(1, 0)].abs() < 1e-12);
    assert!(vectors.right[(1, 0)].abs() < 1e-12);
}

#[test]
fn tiny_non_schur_lower_band_is_not_admitted_as_roundoff() {
    let tiny = 1e-300;
    let real_a = Mat::<f64>::from_fn(3, 3, |i, j| {
        if matches!((i, j), (0, 1) | (1, 2) | (2, 0)) {
            tiny
        } else {
            0.0
        }
    });
    let real_b = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { tiny } else { 0.0 });
    assert!(real_schur_values(&real_a, &real_b).is_err());
    let complex_a = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(real_a[(i, j)], 0.0));
    let complex_b = Mat::<c64>::from_fn(3, 3, |i, j| c64::new(real_b[(i, j)], 0.0));
    assert!(complex_schur_values(&complex_a, &complex_b).is_err());
}
