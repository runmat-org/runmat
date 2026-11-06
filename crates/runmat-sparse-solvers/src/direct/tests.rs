use super::*;
use crate::csr::{CsrMatrix, TripletBuilder};
use crate::scalar::c64;
use crate::testutil::poisson3d;

fn relative_residual<T: Scalar>(a: &CsrMatrix<T>, x: &[T], b: &[T]) -> f64 {
    let mut r = vec![T::zero(); b.len()];
    a.as_ref().residual(x, b, &mut r);
    par::norm2(&r) / par::norm2(b)
}

/// Dense Gaussian elimination reference (row-major), partial pivoting.
fn dense_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| m[i * n + k].abs().partial_cmp(&m[j * n + k].abs()).unwrap())
            .unwrap();
        for c in 0..n {
            m.swap(k * n + c, p * n + c);
        }
        x.swap(k, p);
        for i in k + 1..n {
            let f = m[i * n + k] / m[k * n + k];
            for c in k..n {
                m[i * n + c] -= f * m[k * n + c];
            }
            x[i] -= f * x[k];
        }
    }
    for k in (0..n).rev() {
        let mut s = x[k];
        for c in k + 1..n {
            s -= m[k * n + c] * x[c];
        }
        x[k] = s / m[k * n + k];
    }
    x
}

#[test]
fn cholesky_matches_dense_reference_with_every_ordering() {
    let a = poisson3d::<f64>(6, 5, 4);
    let n = a.nrows();
    let b = par::seeded_vector::<f64>(n, 1);
    let reference = dense_solve(&a.as_ref().to_dense(), &b, n);
    for ordering in [
        Ordering::Amd,
        Ordering::NestedDissection,
        Ordering::Identity,
    ] {
        let options = DirectOptions {
            ordering: ordering.clone(),
            ..DirectOptions::new(FactorKind::Cholesky)
        };
        let solver =
            DirectSolver::<f64>::new(a.as_ref().into(), options, SolveControl::NONE).unwrap();
        let mut x = b.clone();
        solver.solve_in_place(&mut x, 1).unwrap();
        let error = x
            .iter()
            .zip(&reference)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(error < 1e-12, "{ordering:?}: {error}");
        assert_eq!(solver.inertia().unwrap().positive, n);
    }
}

#[test]
fn symbolic_reuse_refactorizes_new_values_and_rejects_new_patterns() {
    let a = poisson3d::<f64>(9, 9, 9);
    let n = a.nrows();
    let options = DirectOptions {
        workspace: DirectWorkspacePolicy::CallerOwned,
        ..DirectOptions::new(FactorKind::Cholesky)
    };
    let symbolic =
        SymbolicFactorization::analyze::<f64>(a.as_ref().into(), &options, SolveControl::NONE)
            .unwrap();
    let estimate = symbolic.estimate();
    assert!(estimate.factor_entries as usize >= a.nnz() / 2);
    let mut solver = DirectSolver::factorize(
        symbolic.clone(),
        a.as_ref().into(),
        options.clone(),
        None,
        SolveControl::NONE,
    )
    .unwrap();
    let shifted = a.clone().with_shifted_diagonal(2.5);
    solver
        .refactorize(shifted.as_ref().into(), SolveControl::NONE)
        .unwrap();
    let b = par::seeded_vector::<f64>(n, 4);
    let mut x = b.clone();
    solver.solve_in_place(&mut x, 1).unwrap();
    assert!(relative_residual(&shifted, &x, &b) < 1e-13);
    let other = poisson3d::<f64>(9, 9, 9).with_identity_rows(&vec![false; n]);
    let mut extended = TripletBuilder::new(n, n);
    for row in 0..n {
        for (&c, &v) in other
            .as_ref()
            .row(row)
            .0
            .iter()
            .zip(other.as_ref().row(row).1)
        {
            extended.push(row, c, v);
        }
    }
    extended.push(0, n - 1, 0.0);
    extended.push(n - 1, 0, 0.0);
    let extended = extended.build();
    assert_eq!(
        solver
            .refactorize(extended.as_ref().into(), SolveControl::NONE)
            .unwrap_err(),
        SolverError::PatternMismatch
    );
    // Budget refusal happens before numeric work.
    let refused = DirectSolver::<f64>::factorize(
        symbolic,
        a.as_ref().into(),
        options,
        Some(16),
        SolveControl::NONE,
    )
    .unwrap_err();
    assert!(matches!(refused, SolverError::OverBudget { .. }));
}

#[test]
fn finite_workspace_policy_quotes_and_executes_same_nonpivoting_factor() {
    let a = poisson3d::<f64>(4, 4, 4);
    let b = vec![1.0; a.nrows()];
    for kind in [FactorKind::Cholesky, FactorKind::Ldlt] {
        let managed = DirectOptions::new(kind);
        let managed_symbolic =
            SymbolicFactorization::analyze::<f64>(a.as_ref().into(), &managed, SolveControl::NONE)
                .unwrap();
        assert!(matches!(
            DirectSolver::<f64>::factorize(
                managed_symbolic,
                a.as_ref().into(),
                managed,
                Some(u64::MAX),
                SolveControl::NONE
            ),
            Err(SolverError::Unsupported(_))
        ));
        let options = DirectOptions {
            workspace: DirectWorkspacePolicy::CallerOwned,
            ..DirectOptions::new(kind)
        };
        let symbolic =
            SymbolicFactorization::analyze::<f64>(a.as_ref().into(), &options, SolveControl::NONE)
                .unwrap();
        assert_eq!(
            symbolic.workspace_policy(),
            DirectWorkspacePolicy::CallerOwned
        );
        assert!(
            matches!(&symbolic.inner, SymbolicInner::Cholesky(inner) if matches!(inner.raw(), faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Simplicial(_)))
        );
        let quote = symbolic.estimate().peak_bytes();
        let numeric_started = std::sync::atomic::AtomicBool::new(false);
        let observe = |progress: Progress| {
            if progress.phase == ProgressPhase::NumericFactorization {
                numeric_started.store(true, std::sync::atomic::Ordering::Release);
            }
        };
        assert!(matches!(
            DirectSolver::<f64>::factorize(
                symbolic.clone(),
                a.as_ref().into(),
                options.clone(),
                Some(quote - 1),
                SolveControl {
                    progress: Some(&observe),
                    ..SolveControl::NONE
                }
            ),
            Err(SolverError::OverBudget { .. })
        ));
        assert!(!numeric_started.load(std::sync::atomic::Ordering::Acquire));
        let solver = DirectSolver::<f64>::factorize(
            symbolic.clone(),
            a.as_ref().into(),
            options.clone(),
            Some(quote),
            SolveControl::NONE,
        )
        .unwrap();
        let mut x = b.clone();
        solver.solve_in_place(&mut x, 1).unwrap();
        assert!(relative_residual(&a, &x, &b) < 1e-12);
        let mut contradicted = options;
        contradicted.workspace = DirectWorkspacePolicy::BackendManaged;
        assert!(matches!(
            DirectSolver::<f64>::factorize(
                symbolic,
                a.as_ref().into(),
                contradicted,
                None,
                SolveControl::NONE
            ),
            Err(SolverError::Unsupported(_))
        ));
    }
    for kind in [FactorKind::BunchKaufman, FactorKind::Lu] {
        let options = DirectOptions {
            workspace: DirectWorkspacePolicy::CallerOwned,
            ..DirectOptions::new(kind)
        };
        assert!(matches!(
            SymbolicFactorization::analyze::<f64>(a.as_ref().into(), &options, SolveControl::NONE),
            Err(SolverError::Unsupported(_))
        ));
        let managed = DirectOptions::new(kind);
        let symbolic =
            SymbolicFactorization::analyze::<f64>(a.as_ref().into(), &managed, SolveControl::NONE)
                .unwrap();
        assert!(matches!(
            DirectSolver::<f64>::factorize(
                symbolic,
                a.as_ref().into(),
                managed,
                Some(u64::MAX),
                SolveControl::NONE
            ),
            Err(SolverError::Unsupported(_))
        ));
    }
}

#[test]
fn indefinite_shifted_laplacian_bunch_kaufman_inertia_counts_eigenvalues() {
    // 1D Laplacian eigenvalues 2 - 2cos(kπ/(n+1)) are known: count those below the shift.
    let n = 200;
    let mut t = TripletBuilder::new(n, n);
    for i in 0..n {
        t.push(i, i, 2.0);
        if i > 0 {
            t.push(i, i - 1, -1.0);
        }
        if i + 1 < n {
            t.push(i, i + 1, -1.0);
        }
    }
    let a = t.build();
    let shift = 1.234;
    let shifted = a.clone().with_shifted_diagonal(-shift);
    let expected = (1..=n)
        .filter(|&k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos() < shift)
        .count();
    for kind in [FactorKind::BunchKaufman, FactorKind::Ldlt] {
        let solver = DirectSolver::<f64>::new(
            shifted.as_ref().into(),
            DirectOptions {
                // Exercise the stagnation boundary instead of stopping at the first
                // already-small residual. The report must describe the returned vector.
                refinement_tolerance: 0.0,
                ..DirectOptions::new(kind)
            },
            SolveControl::NONE,
        )
        .unwrap();
        let inertia = solver.inertia().unwrap();
        assert_eq!(inertia.negative, expected, "{kind:?}");
        let b = par::seeded_vector::<f64>(n, 2);
        let mut x = vec![0.0; n];
        let report = solver
            .solve_refined(shifted.as_ref().into(), &b, &mut x, 1)
            .unwrap();
        assert!(report.relative_residual < 1e-12, "{kind:?} {report:?}");
        assert_eq!(
            report.relative_residual,
            relative_residual(&shifted, &x, &b),
            "{kind:?}: the report must certify the returned solution"
        );
    }
}

#[test]
fn saddle_point_ldlt_with_signed_regularization() {
    // [K Bᵀ; B 0] with K SPD and B full row rank.
    let k = poisson3d::<f64>(5, 5, 2);
    let n = k.nrows();
    let m = 7;
    let mut t = TripletBuilder::new(n + m, n + m);
    for row in 0..n {
        for (&c, &v) in k.as_ref().row(row).0.iter().zip(k.as_ref().row(row).1) {
            t.push(row, c, v);
        }
    }
    for j in 0..m {
        let col = j * 5 + 1;
        t.push(n + j, col, 1.0);
        t.push(col, n + j, 1.0);
        t.push(n + j, n + j, 0.0);
    }
    let a = t.build();
    let mut signs = vec![1i8; n];
    signs.extend(std::iter::repeat_n(-1i8, m));
    let options = DirectOptions {
        pivot_signs: Some(signs),
        regularization_delta: 1e-10,
        regularization_epsilon: 1e-13,
        ..DirectOptions::new(FactorKind::Ldlt)
    };
    let solver = DirectSolver::<f64>::new(a.as_ref().into(), options, SolveControl::NONE).unwrap();
    let b = par::seeded_vector::<f64>(n + m, 8);
    let mut x = vec![0.0; n + m];
    let report = solver
        .solve_refined(a.as_ref().into(), &b, &mut x, 1)
        .unwrap();
    assert!(report.relative_residual < 1e-12, "{report:?}");
    let inertia = solver.inertia().unwrap();
    assert_eq!((inertia.positive, inertia.negative), (n, m));
}

#[test]
fn lu_solves_real_and_complex_nonsymmetric_from_csr_and_csc() {
    let n1 = 15;
    let n = n1 * n1;
    let mut t = TripletBuilder::new(n, n);
    for j in 0..n1 {
        for i in 0..n1 {
            let r = j * n1 + i;
            t.push(r, r, c64::new(4.0, 0.5));
            if i > 0 {
                t.push(r, r - 1, c64::new(-1.4, 0.1));
            }
            if i + 1 < n1 {
                t.push(r, r + 1, c64::new(-0.6, 0.0));
            }
            if j > 0 {
                t.push(r, r - n1, c64::new(-1.0, -0.2));
            }
            if j + 1 < n1 {
                t.push(r, r + n1, c64::new(-1.0, 0.0));
            }
        }
    }
    let a = t.build();
    let b = par::seeded_vector::<c64>(n, 6);
    let solver = DirectSolver::<c64>::new(
        a.as_ref().into(),
        DirectOptions::new(FactorKind::Lu),
        SolveControl::NONE,
    )
    .unwrap();
    let mut x = b.clone();
    solver.solve_in_place(&mut x, 1).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-13);
    // Same matrix supplied as CSC arrays.
    let at = a.as_ref().transpose(false);
    let csc = CscRef::new_unchecked(n, n, at.row_ptr(), at.col_idx(), at.values());
    let solver = DirectSolver::<c64>::new(
        csc.into(),
        DirectOptions::new(FactorKind::Lu),
        SolveControl::NONE,
    )
    .unwrap();
    let mut y = b.clone();
    solver.solve_in_place(&mut y, 1).unwrap();
    assert!(relative_residual(&a, &y, &b) < 1e-13);
    // Real LU, two right-hand sides.
    let real = poisson3d::<f64>(5, 6, 7);
    let nr = real.nrows();
    let solver = DirectSolver::<f64>::new(
        real.as_ref().into(),
        DirectOptions::new(FactorKind::Lu),
        SolveControl::NONE,
    )
    .unwrap();
    let rhs = par::seeded_vector::<f64>(2 * nr, 3);
    let mut sol = rhs.clone();
    solver.solve_in_place(&mut sol, 2).unwrap();
    for c in 0..2 {
        assert!(
            relative_residual(
                &real,
                &sol[c * nr..(c + 1) * nr],
                &rhs[c * nr..(c + 1) * nr]
            ) < 1e-13
        );
    }
}

#[test]
fn hermitian_cholesky_from_csr_input() {
    let base = poisson3d::<f64>(5, 5, 5);
    let mut t = TripletBuilder::new(base.nrows(), base.nrows());
    for row in 0..base.nrows() {
        let (c, v) = base.as_ref().row(row);
        for (&col, &val) in c.iter().zip(v) {
            let imag = if col > row {
                0.3
            } else if col < row {
                -0.3
            } else {
                0.0
            };
            t.push(row, col, c64::new(val, imag));
        }
    }
    let a = t.build();
    let b = par::seeded_vector::<c64>(a.nrows(), 12);
    let solver = DirectSolver::<c64>::new(
        a.as_ref().into(),
        DirectOptions::new(FactorKind::Cholesky),
        SolveControl::NONE,
    )
    .unwrap();
    let mut x = b.clone();
    solver.solve_in_place(&mut x, 1).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-13);
}

#[test]
fn least_squares_qr_matches_normal_equations() {
    let m = 40;
    let n = 12;
    let mut t = TripletBuilder::new(m, n);
    for i in 0..m {
        t.push(i, i % n, 1.0 + i as f64 * 0.1);
        t.push(i, (i * 7 + 3) % n, -0.5);
    }
    let a = t.build();
    let b = par::seeded_vector::<f64>(m, 13);
    let qr = LeastSquaresQr::<f64>::from_csr(&a, SolveControl::NONE).unwrap();
    let x = qr.solve(&b, 1).unwrap();
    // Normal equations residual Aᵀ(b - A x) ≈ 0.
    let mut ax = vec![0.0; m];
    a.as_ref().spmv(&x, &mut ax);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut atr = vec![0.0; n];
    a.as_ref().spmv_transpose(&r, &mut atr, false);
    assert!(par::norm2(&atr) < 1e-10 * par::norm2(&b));
}

#[test]
fn norm_estimates_match_dense_references() {
    // Nonsymmetric tridiagonal with known inverse norm (dense reference).
    let n = 60;
    let mut t = TripletBuilder::new(n, n);
    for i in 0..n {
        t.push(i, i, 3.0 + (i % 5) as f64);
        if i > 0 {
            t.push(i, i - 1, -1.2);
        }
        if i + 1 < n {
            t.push(i, i + 1, -0.7);
        }
    }
    let a = t.build();
    let solver = DirectSolver::<f64>::new(
        a.as_ref().into(),
        DirectOptions::new(FactorKind::Lu),
        SolveControl::NONE,
    )
    .unwrap();
    // Dense inverse 1-norm.
    let dense = a.as_ref().to_dense();
    let mut exact = 0.0f64;
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let column = dense_solve(&dense, &e, n);
        exact = exact.max(column.iter().map(|v| v.abs()).sum());
    }
    let estimate = solver.estimate_inverse_norm1().unwrap();
    assert!(
        estimate <= exact * (1.0 + 1e-12) && estimate >= 0.3 * exact,
        "{estimate} vs {exact}"
    );
    let defect = solver
        .estimate_right_inverse_defect1(a.as_ref().into())
        .unwrap();
    assert!(defect < 1e-12, "{defect}");
    // Transpose solve.
    let b = par::seeded_vector::<f64>(n, 3);
    let mut x = b.clone();
    solver.solve_transpose_in_place(&mut x, 1).unwrap();
    let at = a.as_ref().transpose(false);
    assert!(relative_residual(&at, &x, &b) < 1e-13);
}

#[test]
fn count_only_estimate_equals_the_simplicial_analysis() {
    for (nx, ny, nz) in [(3, 3, 3), (6, 5, 4), (12, 12, 12)] {
        let matrix = poisson3d::<f64>(nx, ny, nz);
        for kind in [FactorKind::Cholesky, FactorKind::Ldlt] {
            let options = DirectOptions {
                workspace: DirectWorkspacePolicy::CallerOwned,
                ..DirectOptions::new(kind)
            };
            let n = matrix.nrows();
            let full = SymbolicFactorization::analyze_pattern::<f64>(
                n,
                matrix.row_ptr(),
                matrix.col_idx(),
                &options,
                SolveControl::NONE,
            )
            .unwrap()
            .estimate();
            let counted = SymbolicFactorization::estimate_pattern::<f64>(
                n,
                matrix.row_ptr(),
                matrix.col_idx(),
                &options,
                SolveControl::NONE,
            )
            .unwrap();
            assert_eq!(counted, full, "{nx}x{ny}x{nz} {kind:?}");
        }
    }
}
