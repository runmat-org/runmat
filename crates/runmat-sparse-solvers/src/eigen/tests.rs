use super::*;
use crate::csr::CsrMatrix;
use crate::precond::{Amg, AmgOptions};
use crate::testutil::poisson3d;

fn analytic(nx: usize, ny: usize, nz: usize, count: usize) -> Vec<f64> {
    let e =
        |k: usize, n: usize| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
    let mut all = Vec::new();
    for i in 1..=nx.min(8) {
        for j in 1..=ny.min(8) {
            for k in 1..=nz.min(8) {
                all.push(e(i, nx) + e(j, ny) + e(k, nz));
            }
        }
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all.truncate(count);
    all
}

#[test]
fn shift_invert_block_lanczos_finds_repeated_laplacian_modes_with_sturm_proof() {
    let (nx, ny, nz) = (20, 20, 20);
    let k = poisson3d::<f64>(nx, ny, nz);
    let m = CsrMatrix::<f64>::identity(k.nrows());
    let options = EigenOptions {
        count: 12,
        sturm_check: true,
        ..Default::default()
    };
    let result = modal(k.as_ref(), m.as_ref(), &options, SolveControl::NONE).unwrap();
    assert!(result.converged);
    let expected = analytic(nx, ny, nz, 12);
    for (v, e) in result.values.iter().zip(&expected) {
        assert!((v - e).abs() <= 1e-9 * e, "{v} vs {e}");
    }
    assert!(
        result.residuals.iter().all(|&r| r < 1e-8),
        "{:?}",
        result.residuals
    );
    let sturm = result.sturm.unwrap();
    // The 12th and 13th eigenvalues may coincide (multiplicity); the inertia counts every
    // eigenvalue below the probe, so it may exceed the returned count only through such ties.
    let expected_below = analytic(nx, ny, nz, 40)
        .iter()
        .filter(|&&v| v < sturm.shift)
        .count();
    assert_eq!(sturm.counted, expected_below);
}

#[test]
fn lobpcg_with_amg_matches_lanczos_on_generalized_problem() {
    let (nx, ny, nz) = (14, 14, 14);
    let k = poisson3d::<f64>(nx, ny, nz);
    let n = k.nrows();
    let diag: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.5 * ((i as f64) * 0.37).sin().abs())
        .collect();
    let m = CsrMatrix::from_diagonal(&diag);
    let lanczos = modal(
        k.as_ref(),
        m.as_ref(),
        &EigenOptions {
            count: 6,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    let amg = Amg::new(k.as_ref(), AmgOptions::default(), SolveControl::NONE).unwrap();
    let lob = lobpcg(
        k.as_ref(),
        m.as_ref(),
        Some(&amg),
        &LobpcgOptions {
            count: 6,
            tolerance: 1e-9,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    assert!(lob.converged);
    for (a, b) in lanczos.values.iter().zip(&lob.values) {
        assert!((a - b).abs() <= 1e-8 * a, "{a} vs {b}");
    }
}

#[test]
fn buckling_load_factors_match_modal_equivalent() {
    let k = poisson3d::<f64>(15, 15, 15);
    let n = k.nrows();
    let m = CsrMatrix::<f64>::identity(n);
    // K_G = −M  ⇒  (K − λ M) φ = 0.
    let kg = CsrMatrix::linear_combination(&[(-1.0, m.as_ref())]).unwrap();
    let options = EigenOptions {
        count: 5,
        ..Default::default()
    };
    let b = buckling(k.as_ref(), kg.as_ref(), &options, SolveControl::NONE).unwrap();
    let expected = analytic(15, 15, 15, 5);
    for (v, e) in b.values.iter().zip(&expected) {
        assert!((v - e).abs() <= 1e-9 * e, "{v} vs {e}");
    }
}

#[test]
fn small_problems_use_the_dense_path() {
    let k = poisson3d::<f64>(5, 4, 3);
    let m = CsrMatrix::<f64>::identity(k.nrows());
    let r = modal(
        k.as_ref(),
        m.as_ref(),
        &EigenOptions {
            count: 4,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    let expected = analytic(5, 4, 3, 4);
    for (v, e) in r.values.iter().zip(&expected) {
        assert!((v - e).abs() <= 1e-10 * e);
    }
}

#[test]
fn largest_algebraic_eigenvalues_without_shift_invert() {
    let (nx, ny, nz) = (12, 12, 12);
    let k = poisson3d::<f64>(nx, ny, nz);
    let e =
        |i: usize, n: usize| 2.0 - 2.0 * (i as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
    let mut all = Vec::new();
    for i in 1..=nx {
        for j in 1..=ny {
            for l in 1..=nz {
                all.push(e(i, nx) + e(j, ny) + e(l, nz));
            }
        }
    }
    all.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let result = symmetric_extreme(
        k.as_ref(),
        Selection::LargestAlgebraic,
        &EigenOptions {
            count: 4,
            tolerance: 1e-9,
            max_restarts: 500,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    assert!(result.converged);
    for (v, x) in result.values.iter().zip(&all) {
        assert!((v - x).abs() <= 1e-7 * x, "{v} vs {x}");
    }
}

#[test]
fn modal_working_set_follows_the_requested_count() {
    let options = |count| EigenOptions {
        count,
        ..Default::default()
    };
    let n = 1_000_000;
    let p = DEFAULT_BLOCK_SIZE;
    // Lanczos: basis max(2s + 2p, s + 8p) and its restart copy, four blocks, two copies of
    // the Ritz vectors, four projected matrices of the basis size.
    let basis = options(13).basis_capacity(n);
    assert_eq!(basis, (2 * 13 + 2 * p).max(13 + 8 * p));
    assert_eq!(
        modal_working_value_count(&options(13), n),
        Some(n * (2 * basis + 4 * p + 2 * 13) + 4 * basis * basis)
    );
    assert!(
        modal_working_value_count(&options(13), n).unwrap() * 9
            < modal_working_value_count(&options(256), n).unwrap()
    );
    // Dense fallback on a small pencil: at least the densified pencil and its copies.
    let small = 300;
    assert!(modal_working_value_count(&options(13), small).unwrap() >= 5 * small * small);
    assert_eq!(modal_working_value_count(&options(13), usize::MAX), None);
}

/// A consistent-mass-like SPD pencil on a 3D grid: `M = I + 0.1·L` with `L` the graph
/// Laplacian pattern, so `K` and `M` share a pattern with off-diagonal mass coupling.
fn coupled_mass(k: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = k.nrows();
    let identity = CsrMatrix::<f64>::identity(n);
    CsrMatrix::linear_combination(&[(1.0, identity.as_ref()), (0.02, k.as_ref())]).unwrap()
}

fn dense_eigenvalues(k: &CsrMatrix<f64>, m: &CsrMatrix<f64>) -> Vec<f64> {
    let (values, _) = dense_generalized(k.as_ref(), m.as_ref()).unwrap();
    let mut values = values;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

#[test]
fn factor_free_inertia_counts_every_eigenvalue_below_each_shift() {
    let k = poisson3d::<f64>(9, 8, 7);
    let m = coupled_mass(&k);
    let values = dense_eigenvalues(&k, &m);
    let inertia = PencilInertia::analyze(
        k.as_ref(),
        m.as_ref(),
        &crate::ordering::Ordering::Auto,
        SolveControl::NONE,
    )
    .unwrap();
    for index in [0usize, 1, 7, 40, 200, 503] {
        // Midway between neighbouring eigenvalues: the count is exact.
        let shift = if index == 0 {
            0.5 * values[0]
        } else {
            0.5 * (values[index - 1] + values[index])
        };
        let counted = inertia.count(shift, SolveControl::NONE).unwrap();
        assert_eq!(counted.negative, index, "shift {shift}");
        assert_eq!(counted.negative + counted.positive, k.nrows());
        assert_eq!(counted.zero, 0);
    }
}

#[test]
fn factor_free_inertia_matches_on_elasticity_and_quotes_its_peak_from_the_pattern() {
    let problem = crate::testutil::elasticity::elasticity3d(4, 3, 6);
    let k = problem.matrix;
    let m = coupled_mass(&k);
    let values = dense_eigenvalues(&k, &m);
    let inertia = PencilInertia::analyze(
        k.as_ref(),
        m.as_ref(),
        &crate::ordering::Ordering::Auto,
        SolveControl::NONE,
    )
    .unwrap();
    for index in [1usize, 5, 13, 60] {
        let shift = 0.5 * (values[index - 1] + values[index]);
        assert_eq!(
            inertia.count(shift, SolveControl::NONE).unwrap().negative,
            index
        );
    }
    // The planner's quote from the union pattern alone equals the analyzed peak.
    let estimate = pencil_inertia_estimate(
        k.nrows(),
        m.row_ptr(),
        m.col_idx(),
        &crate::ordering::Ordering::Auto,
    )
    .unwrap();
    assert_eq!(estimate.resident_bytes, inertia.resident_bytes());
    assert!(estimate.flops > 0.0);
}

#[test]
fn sturm_check_runs_after_the_factor_is_released_and_proves_completeness() {
    let k = poisson3d::<f64>(16, 15, 14);
    let m = coupled_mass(&k);
    let options = EigenOptions {
        count: 9,
        sturm_check: true,
        ..Default::default()
    };
    let result = modal(k.as_ref(), m.as_ref(), &options, SolveControl::NONE).unwrap();
    let sturm = result.sturm.expect("modal runs the Sturm check");
    assert!(sturm.complete(), "{sturm:?}");
    assert_eq!(sturm.returned, 9);
}

#[test]
fn lobpcg_modal_path_matches_shift_invert_frequencies_with_its_own_proof() {
    let problem = crate::testutil::elasticity::elasticity3d(5, 4, 10);
    let k = problem.matrix;
    let m = coupled_mass(&k);
    let count = 6;
    let direct = modal(
        k.as_ref(),
        m.as_ref(),
        &EigenOptions {
            count,
            sturm_check: true,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    let iterative = modal_lobpcg(
        k.as_ref(),
        m.as_ref(),
        &IterativeModalOptions {
            lobpcg: LobpcgOptions {
                count,
                guard: 6,
                tolerance: 1e-9,
                measure: ResidualMeasure::RelativeResidual,
                max_iterations: 2000,
            },
            ordering: crate::ordering::Ordering::Auto,
        },
        SolveControl::NONE,
    )
    .unwrap();
    assert!(iterative.converged);
    let sturm = iterative
        .sturm
        .expect("the iterative path proves completeness");
    assert!(sturm.complete(), "{sturm:?}");
    for (a, b) in iterative.values.iter().zip(&direct.values) {
        // Frequencies (√λ) agree to 1e-7 relative.
        assert!((a.sqrt() - b.sqrt()).abs() <= 1e-7 * b.sqrt(), "{a} vs {b}");
    }
}

#[test]
fn lobpcg_modal_path_refuses_an_indefinite_stiffness() {
    let k = poisson3d::<f64>(9, 9, 9).with_shifted_diagonal(-0.5);
    let m = CsrMatrix::<f64>::identity(k.nrows());
    let refused = modal_lobpcg(
        k.as_ref(),
        m.as_ref(),
        &IterativeModalOptions {
            lobpcg: LobpcgOptions {
                count: 3,
                guard: 3,
                tolerance: 1e-8,
                measure: ResidualMeasure::RelativeResidual,
                max_iterations: 50,
            },
            ordering: crate::ordering::Ordering::Auto,
        },
        SolveControl::NONE,
    );
    // Either the preconditioner, the iteration or the zero-shift inertia refuses; modes are
    // never returned for an indefinite stiffness.
    assert!(refused.is_err(), "{refused:?}");
}

/// A slender clamped elastic beam (bending dominated) at a tight tolerance: every multigrid
/// variant reaches it once the search directions are M-orthonormalized, and the rigid-body
/// near-nullspace (which broke down without that) needs the fewest steps.
#[test]
fn lobpcg_reaches_a_tight_tolerance_on_a_slender_beam_with_each_multigrid_variant() {
    use crate::precond::rigid_body_modes;
    let p = crate::testutil::elasticity::elasticity3d(4, 4, 60);
    let k = p.matrix;
    let m = CsrMatrix::<f64>::identity(k.nrows());
    let options = LobpcgOptions {
        count: 8,
        guard: 6,
        tolerance: 5e-9,
        measure: ResidualMeasure::RelativeResidual,
        max_iterations: 400,
    };
    let blocked = AmgOptions {
        block_size: 3,
        strength_threshold: 0.02,
        coarse_size: 200,
        ..Default::default()
    };
    let variants = [
        AmgOptions {
            coarse_size: 200,
            ..Default::default()
        },
        blocked.clone(),
        AmgOptions {
            near_nullspace: Some(rigid_body_modes(&p.coordinates, 3)),
            ..blocked
        },
    ];
    let mut steps = Vec::new();
    let mut reference: Option<Vec<f64>> = None;
    for amg_options in variants {
        let amg = Amg::new(k.as_ref(), amg_options, SolveControl::NONE).unwrap();
        let r = lobpcg(
            k.as_ref(),
            m.as_ref(),
            Some(&amg),
            &options,
            SolveControl::NONE,
        )
        .unwrap();
        assert!(
            r.converged,
            "{} steps, residuals {:?}",
            r.restarts, r.residuals
        );
        if let Some(reference) = &reference {
            for (a, b) in r.values.iter().zip(reference) {
                assert!((a - b).abs() <= 1e-7 * b, "{a} vs {b}");
            }
        } else {
            reference = Some(r.values.clone());
        }
        steps.push(r.restarts);
    }
    assert!(steps[2] < steps[1] && steps[1] < steps[0], "{steps:?}");
}

/// The 1-D Laplacian pencil `K x = λ x` has the exact pairs `λ_j = 2 − 2cos(jπ/(n+1))`,
/// `x_j(i) = sin(ijπ/(n+1))`. Rounding the exact vector to double precision leaves a relative
/// residual near `u · λ_max/λ_1`, above 1e-8 at this size, while its backward error stays at the
/// unit roundoff; a mixture of the two lowest modes is rejected by the backward error.
#[test]
fn backward_error_qualifies_an_exact_pair_the_relative_residual_rejects() {
    let n = 40_000;
    let mut builder = crate::csr::TripletBuilder::new(n, n);
    for i in 0..n {
        builder.push(i, i, 2.0);
        if i + 1 < n {
            builder.push(i, i + 1, -1.0);
            builder.push(i + 1, i, -1.0);
        }
    }
    let k = builder.build();
    let identity = CsrMatrix::<f64>::identity(n);
    let mode = |j: usize| -> (f64, Vec<f64>) {
        let angle = j as f64 * std::f64::consts::PI / (n + 1) as f64;
        let scale = (2.0 / (n + 1) as f64).sqrt();
        (
            2.0 - 2.0 * angle.cos(),
            (1..=n).map(|i| scale * (i as f64 * angle).sin()).collect(),
        )
    };
    let (lambda_1, x_1) = mode(1);
    let (lambda_2, x_2) = mode(2);
    let (relative, backward) =
        super::true_residuals(k.as_ref(), identity.as_ref(), &[lambda_1], &x_1);
    assert!(relative[0] > 1.0e-8, "relative residual {}", relative[0]);
    assert!(backward[0] < 1.0e-15, "backward error {}", backward[0]);
    // A wrong eigenvector: an equal mixture of the two lowest modes with its Rayleigh quotient.
    let mixed: Vec<f64> = x_1
        .iter()
        .zip(&x_2)
        .map(|(a, b)| (a + b) / 2f64.sqrt())
        .collect();
    let quotient = 0.5 * (lambda_1 + lambda_2);
    let (_, backward_mixed) =
        super::true_residuals(k.as_ref(), identity.as_ref(), &[quotient], &mixed);
    assert!(backward_mixed[0] > 1.0e-10, "mixed backward error {}", backward_mixed[0]);
}

/// LOBPCG stopping on the backward error reaches the same frequencies as shift-invert, and every
/// returned pair carries both measures.
#[test]
fn lobpcg_stops_on_the_backward_error_and_reports_both_measures() {
    let problem = crate::testutil::elasticity::elasticity3d(4, 4, 12);
    let k = problem.matrix;
    let m = coupled_mass(&k);
    let count = 5;
    let reference = modal(
        k.as_ref(),
        m.as_ref(),
        &EigenOptions {
            count,
            shift: Some(0.0),
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    let amg = crate::precond::Amg::new(
        k.as_ref(),
        crate::precond::AmgOptions::default(),
        SolveControl::NONE,
    )
    .unwrap();
    let tolerance = 1.0e-11;
    let result = lobpcg(
        k.as_ref(),
        m.as_ref(),
        Some(&amg),
        &LobpcgOptions {
            count,
            guard: 6,
            tolerance,
            measure: ResidualMeasure::BackwardError,
            max_iterations: 2000,
        },
        SolveControl::NONE,
    )
    .unwrap();
    assert!(result.converged);
    assert_eq!(result.backward_errors.len(), count);
    assert_eq!(result.residuals.len(), count);
    for (index, (&value, &expected)) in result.values.iter().zip(&reference.values).enumerate() {
        assert!(result.backward_errors[index] <= tolerance);
        assert!(result.backward_errors[index] <= result.residuals[index]);
        assert!(
            (value - expected).abs() <= 1.0e-8 * expected,
            "mode {index}: {value} vs {expected}"
        );
    }
    assert!(reference.backward_errors.iter().all(|&eta| eta < 1.0e-14));
}
