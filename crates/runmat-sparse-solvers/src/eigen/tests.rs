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
