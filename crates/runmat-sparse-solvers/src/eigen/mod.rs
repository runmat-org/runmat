//! Generalized symmetric eigensolvers. Their eigen iterations and projected
//! eigendecompositions are explicitly excluded from rsua-1 in 0.7.0; measured
//! child matrix/vector/direct-solve kernels retain their separate charges.
//!
//! - [`modal`]: `K x = λ M x` (stiffness/mass, also prestressed `K + K_G`), eigenvalues nearest
//!   above a shift via shift-invert thick-restart block Lanczos on `(K − σM)⁻¹ M` with a sparse
//!   Cholesky (or Bunch–Kaufman for interior shifts), plus an optional Sturm-sequence check that
//!   proves no eigenvalue below the highest returned one was missed.
//! - [`buckling`]: `(K + λ K_G) φ = 0`, smallest positive load factors via Lanczos on
//!   `K⁻¹(−K_G)` in the `K` inner product.
//! - [`lobpcg`]: preconditioned block solver (e.g. with AMG) for the lowest modes when a
//!   factorization does not fit in memory.

mod blas;
mod lanczos;
mod lobpcg;

pub use lanczos::Selection;
pub use lobpcg::{lobpcg, LobpcgOptions};

use crate::control::SolveControl;
use crate::csr::{CsrMatrix, CsrRef};
use crate::dense::{cholesky_in_place, symmetric_eigen};
use crate::direct::{DirectOptions, DirectSolver, FactorEstimate, FactorKind};
use crate::error::{SolverError, SolverResult};
use crate::ordering::Ordering;
use crate::par;
use lanczos::{thick_restart_block_lanczos, LanczosConfig};

/// Options shared by the Lanczos drivers.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenOptions {
    pub count: usize,
    /// Spectral shift `σ` (`None`: automatic, just below the spectrum).
    pub shift: Option<f64>,
    pub block_size: usize,
    /// Basis capacity (`None`: `max(2·count + 2·block, count + 8·block)`).
    pub max_basis: Option<usize>,
    /// Relative accuracy of the eigenvalues.
    pub tolerance: f64,
    pub max_restarts: usize,
    /// Factor `K − σ'M` above the highest returned eigenvalue and compare its inertia with the
    /// returned count.
    pub sturm_check: bool,
    pub ordering: Ordering,
}

impl Default for EigenOptions {
    fn default() -> Self {
        Self {
            count: 10,
            shift: None,
            block_size: 6,
            max_basis: None,
            tolerance: 1e-10,
            max_restarts: 200,
            sturm_check: false,
            ordering: Ordering::Auto,
        }
    }
}

/// Inertia-based completeness proof.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SturmCheck {
    pub shift: f64,
    /// Eigenvalues below `shift` according to the inertia of `K − shift·M`.
    pub counted: usize,
    /// Returned eigenvalues below `shift`.
    pub returned: usize,
}

impl SturmCheck {
    pub fn complete(&self) -> bool {
        self.counted == self.returned
    }
}

/// Eigenpairs, ascending by eigenvalue.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenResult {
    pub values: Vec<f64>,
    /// `n × values.len()` column-major, normalized in the problem's inner product.
    pub vectors: Vec<f64>,
    /// True relative residuals `‖A x − λ B x‖ / (‖A x‖ + |λ| ‖B x‖)`; for [`buckling`], the
    /// relative load-factor error bound `‖K φ − λ (−K_G) φ‖_{K⁻¹} / ‖φ‖_K`.
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub restarts: usize,
    pub operator_applications: usize,
    pub shift: f64,
    pub factorization: Option<FactorEstimate>,
    pub factor_kind: Option<FactorKind>,
    pub sturm: Option<SturmCheck>,
}

fn basis_size(options: &EigenOptions, n: usize) -> usize {
    let p = options.block_size.max(1);
    let wanted = options
        .max_basis
        .unwrap_or_else(|| (2 * options.count + 2 * p).max(options.count + 8 * p));
    wanted.min(n)
}

/// Dense fallback for small problems: `A x = λ B x` with `B` SPD after an optional shift
/// transformation. Returns eigenvalues ascending (all of them) and B-normalized vectors.
fn dense_generalized(a: CsrRef<'_, f64>, b: CsrRef<'_, f64>) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    let n = a.nrows();
    let dense = |m: CsrRef<'_, f64>| {
        let mut d = vec![0.0; n * n];
        for row in 0..n {
            let (cols, vals) = m.row(row);
            for (&c, &v) in cols.iter().zip(vals) {
                d[c * n + row] = v;
            }
        }
        d
    };
    let (ad, mut l) = (dense(a), dense(b));
    cholesky_in_place(&mut l, n)?;
    // C = L⁻¹ A L⁻ᵀ.
    let lower = |i: usize, j: usize| if i >= j { l[j * n + i] } else { 0.0 };
    let mut c = ad.clone();
    // Solve L Y = A (columns), then L Z = Yᵀ.
    for col in 0..n {
        for i in 0..n {
            let mut s = c[col * n + i];
            for k in 0..i {
                s -= lower(i, k) * c[col * n + k];
            }
            c[col * n + i] = s / lower(i, i);
        }
    }
    let mut ct = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            ct[i * n + j] = c[j * n + i];
        }
    }
    for col in 0..n {
        for i in 0..n {
            let mut s = ct[col * n + i];
            for k in 0..i {
                s -= lower(i, k) * ct[col * n + k];
            }
            ct[col * n + i] = s / lower(i, i);
        }
    }
    let (values, z) = symmetric_eigen(&ct, n)?;
    // x = L⁻ᵀ z.
    let mut x = z;
    for col in 0..n {
        for i in (0..n).rev() {
            let mut s = x[col * n + i];
            for k in i + 1..n {
                s -= lower(k, i) * x[col * n + k];
            }
            x[col * n + i] = s / lower(i, i);
        }
    }
    Ok((values, x))
}

fn true_residuals(
    a: CsrRef<'_, f64>,
    b: CsrRef<'_, f64>,
    values: &[f64],
    vectors: &[f64],
) -> Vec<f64> {
    let n = a.nrows();
    let mut ax = vec![0.0; n];
    let mut bx = vec![0.0; n];
    values
        .iter()
        .enumerate()
        .map(|(i, &lambda)| {
            let x = &vectors[i * n..(i + 1) * n];
            a.spmv(x, &mut ax);
            b.spmv(x, &mut bx);
            let (na, nb) = (par::norm2(&ax), par::norm2(&bx));
            par::axpy(-lambda, &bx, &mut ax);
            par::norm2(&ax) / (na + lambda.abs() * nb).max(f64::MIN_POSITIVE)
        })
        .collect()
}

/// `K x = λ M x`: the `count` eigenvalues nearest above the shift (lowest modes by default).
pub fn modal(
    k: CsrRef<'_, f64>,
    m: CsrRef<'_, f64>,
    options: &EigenOptions,
    control: SolveControl<'_>,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    if k.ncols() != n || m.nrows() != n || m.ncols() != n {
        return Err(SolverError::Dimension(
            "modal operators must be square and equal".into(),
        ));
    }
    let count = options.count.min(n);
    let p = options.block_size.max(1);
    // Small problems: dense.
    if n <= 3 * basis_size(options, n).max(count + 2 * p) || n <= 400 {
        return dense_modal(k, m, options, options.shift.unwrap_or(0.0));
    }
    // Factor K − σM: Cholesky when σ is below the spectrum, Bunch–Kaufman otherwise.
    let diagonal_ratio = {
        let (dk, dm) = (k.diagonal(), m.diagonal());
        let sk: f64 = dk.iter().map(|v| v.abs()).sum();
        let sm: f64 = dm
            .iter()
            .map(|v| v.abs())
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        sk / sm
    };
    let candidates: Vec<(f64, FactorKind)> = match options.shift {
        Some(shift) => vec![
            (shift, FactorKind::Cholesky),
            (shift, FactorKind::BunchKaufman),
        ],
        None => {
            let mut c = vec![(0.0, FactorKind::Cholesky)];
            for e in [-6, -5, -4, -3] {
                c.push((-diagonal_ratio * 10f64.powi(e), FactorKind::Cholesky));
            }
            c
        }
    };
    let mut chosen = None;
    let mut last_error = None;
    for (shift, kind) in candidates {
        control.check()?;
        let shifted = CsrMatrix::linear_combination(&[(1.0, k), (-shift, m)])?;
        let direct = DirectOptions {
            ordering: options.ordering.clone(),
            ..DirectOptions::new(kind)
        };
        match DirectSolver::<f64>::new(shifted.as_ref().into(), direct, control) {
            Ok(solver) => {
                chosen = Some((shift, solver));
                break;
            }
            Err(
                error @ (SolverError::NotPositiveDefinite { .. } | SolverError::Singular { .. }),
            ) => {
                last_error = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    let Some((shift, solver)) = chosen else {
        return Err(last_error.unwrap_or(SolverError::Breakdown("no admissible shift".into())));
    };
    modal_with_factor(k, m, &solver, shift, options, control)
}

/// Shift-invert Lanczos with a caller-provided factorization of `K − shift·M` (for example the
/// Cholesky factor that proved `K` coercive, with `shift = 0`).
pub fn modal_with_factor(
    k: CsrRef<'_, f64>,
    m: CsrRef<'_, f64>,
    solver: &DirectSolver<f64>,
    shift: f64,
    options: &EigenOptions,
    control: SolveControl<'_>,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    if solver.dimension() != n || m.nrows() != n {
        return Err(SolverError::Dimension(
            "factorization does not match the pencil".into(),
        ));
    }
    let count = options.count.min(n);
    if n <= 400 || n < count + 4 * options.block_size.max(1) {
        return dense_modal(k, m, options, shift);
    }
    let p = options.block_size.max(1);
    let below_shift_known_empty = solver.kind() == FactorKind::Cholesky;
    let config = LanczosConfig {
        n,
        nev: count,
        block: p,
        max_basis: basis_size(options, n),
        tolerance: options.tolerance,
        max_restarts: options.max_restarts,
        selection: lanczos::Selection::LargestMagnitude,
        seed: 0x6d6f_6461_6c00_0001,
    };
    let mut mx = vec![0.0; n * p];
    let mut apply_op = |x: &[f64], y: &mut [f64], cols: usize| -> SolverResult<()> {
        blas::spmm(m, x, &mut mx[..n * cols], cols);
        y.copy_from_slice(&mx[..n * cols]);
        solver.solve_in_place(y, cols)
    };
    let apply_b = |x: &[f64], y: &mut [f64], cols: usize| blas::spmm(m, x, y, cols);
    let output = thick_restart_block_lanczos(&config, &mut apply_op, &apply_b, control)?;
    // λ = σ + 1/θ, ascending.
    let mut order: Vec<usize> = (0..output.theta.len()).collect();
    let lambda = |i: usize| shift + 1.0 / output.theta[i];
    order.sort_by(|&a, &b| lambda(a).partial_cmp(&lambda(b)).unwrap());
    let values: Vec<f64> = order.iter().map(|&i| lambda(i)).collect();
    let mut vectors = vec![0.0; n * values.len()];
    for (c, &i) in order.iter().enumerate() {
        vectors[c * n..(c + 1) * n].copy_from_slice(&output.vectors[i * n..(i + 1) * n]);
    }
    let residuals = true_residuals(k, m, &values, &vectors);
    let sturm = if options.sturm_check && below_shift_known_empty && !values.is_empty() {
        Some(sturm_count(k, m, &values, options, control)?)
    } else {
        None
    };
    Ok(EigenResult {
        converged: output.converged >= count,
        values,
        vectors,
        residuals,
        restarts: output.restarts,
        operator_applications: output.operator_applications,
        shift,
        factorization: Some(solver.symbolic().estimate()),
        factor_kind: Some(solver.kind()),
        sturm,
    })
}

/// Dense generalized solve for small pencils (all eigenvalues, then the `count` nearest above
/// `shift`).
fn dense_modal(
    k: CsrRef<'_, f64>,
    m: CsrRef<'_, f64>,
    options: &EigenOptions,
    shift: f64,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    let count = options.count.min(n);
    let shifted = CsrMatrix::linear_combination(&[(1.0, k), (-shift, m)])?;
    // (K - σM) x = (λ - σ) M x  ⇔  M x = μ (K - σM) x with μ = 1/(λ-σ).
    let (mu, x) = dense_generalized(m, shifted.as_ref()).or_else(|_| {
        // M singular: use the SPD shifted stiffness as the B-matrix instead.
        dense_generalized(shifted.as_ref(), m)
            .map(|(lam, x)| (lam.iter().map(|l| 1.0 / l).collect(), x))
    })?;
    let mut pairs: Vec<(f64, usize)> = mu
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite() && v.abs() > 0.0)
        .map(|(i, &v)| (shift + 1.0 / v, i))
        .filter(|(lambda, _)| *lambda >= shift || options.shift.is_some())
        .collect();
    pairs.sort_by(|a, b| {
        (a.0 - shift)
            .abs()
            .partial_cmp(&(b.0 - shift).abs())
            .unwrap()
    });
    pairs.truncate(count);
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let values: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let mut vectors = vec![0.0; n * values.len()];
    let mut mx = vec![0.0; n];
    for (c, &(_, i)) in pairs.iter().enumerate() {
        let x = &x[i * n..(i + 1) * n];
        m.spmv(x, &mut mx);
        let norm = par::dot(x, &mx).abs().sqrt().max(f64::MIN_POSITIVE);
        for j in 0..n {
            vectors[c * n + j] = x[j] / norm;
        }
    }
    let residuals = true_residuals(k, m, &values, &vectors);
    Ok(EigenResult {
        converged: values.len() == count,
        values,
        vectors,
        residuals,
        restarts: 0,
        operator_applications: 0,
        shift,
        factorization: None,
        factor_kind: None,
        sturm: None,
    })
}

fn sturm_count(
    k: CsrRef<'_, f64>,
    m: CsrRef<'_, f64>,
    values: &[f64],
    options: &EigenOptions,
    control: SolveControl<'_>,
) -> SolverResult<SturmCheck> {
    let top = *values.last().unwrap();
    let probe = top + (top.abs() * (100.0 * options.tolerance).max(1e-8)).max(f64::MIN_POSITIVE);
    let shifted = CsrMatrix::linear_combination(&[(1.0, k), (-probe, m)])?;
    let solver = DirectSolver::<f64>::new(
        shifted.as_ref().into(),
        DirectOptions {
            ordering: options.ordering.clone(),
            ..DirectOptions::new(FactorKind::BunchKaufman)
        },
        control,
    )?;
    let inertia = solver.inertia().expect("symmetric factorization");
    Ok(SturmCheck {
        shift: probe,
        counted: inertia.negative,
        returned: values.iter().filter(|&&v| v < probe).count(),
    })
}

/// Linear buckling `(K + λ K_G) φ = 0`: the `count` smallest positive load factors.
pub fn buckling(
    k: CsrRef<'_, f64>,
    kg: CsrRef<'_, f64>,
    options: &EigenOptions,
    control: SolveControl<'_>,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    if k.ncols() != n || kg.nrows() != n || kg.ncols() != n {
        return Err(SolverError::Dimension(
            "buckling operators must be square and equal".into(),
        ));
    }
    let count = options.count.min(n);
    let p = options.block_size.max(1);
    // The Cholesky factor of K is the Lanczos operator and the norm of the residual bound, so
    // the dense path for small pencils factors K as well.
    let solver = DirectSolver::<f64>::new(
        k.into(),
        DirectOptions {
            ordering: options.ordering.clone(),
            ..DirectOptions::new(FactorKind::Cholesky)
        },
        control,
    )?;
    let dense = n <= 400 || n <= 3 * basis_size(options, n).max(count + 2 * p);
    solve_buckling(k, kg, &solver, options, control, dense)
}

/// Buckling with a caller-provided Cholesky factorization of `K`.
pub fn buckling_with_factor(
    k: CsrRef<'_, f64>,
    kg: CsrRef<'_, f64>,
    solver: &DirectSolver<f64>,
    options: &EigenOptions,
    control: SolveControl<'_>,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    let count = options.count.min(n);
    let p = options.block_size.max(1);
    solve_buckling(
        k,
        kg,
        solver,
        options,
        control,
        n <= 400 || n < count + 4 * p,
    )
}

fn solve_buckling(
    k: CsrRef<'_, f64>,
    kg: CsrRef<'_, f64>,
    solver: &DirectSolver<f64>,
    options: &EigenOptions,
    control: SolveControl<'_>,
    dense: bool,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    if solver.dimension() != n || kg.nrows() != n || kg.ncols() != n {
        return Err(SolverError::Dimension(
            "factorization does not match the pencil".into(),
        ));
    }
    let count = options.count.min(n);
    let p = options.block_size.max(1);
    let negative_kg = CsrMatrix::linear_combination(&[(-1.0, kg)])?;
    let (values, vectors, lanczos_converged, restarts, operator_applications) = if dense {
        let (values, vectors) = dense_buckling(k, negative_kg.as_ref(), count)?;
        (values, vectors, true, 0, 0)
    } else {
        let config = LanczosConfig {
            n,
            nev: count,
            block: p,
            max_basis: basis_size(options, n),
            tolerance: options.tolerance,
            max_restarts: options.max_restarts,
            selection: lanczos::Selection::LargestAlgebraic,
            seed: 0x6275_636b_6c00_0001,
        };
        let mut apply_op = |x: &[f64], y: &mut [f64], cols: usize| -> SolverResult<()> {
            blas::spmm(negative_kg.as_ref(), x, y, cols);
            solver.solve_in_place(y, cols)
        };
        let apply_b = |x: &[f64], y: &mut [f64], cols: usize| blas::spmm(k, x, y, cols);
        let output = thick_restart_block_lanczos(&config, &mut apply_op, &apply_b, control)?;
        let mut pairs: Vec<(f64, usize)> = output
            .theta
            .iter()
            .enumerate()
            .filter(|(_, &t)| t > 0.0)
            .map(|(i, &t)| (1.0 / t, i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let values: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let mut vectors = vec![0.0; n * values.len()];
        for (c, &(_, i)) in pairs.iter().enumerate() {
            vectors[c * n..(c + 1) * n].copy_from_slice(&output.vectors[i * n..(i + 1) * n]);
        }
        (
            values,
            vectors,
            output.converged >= count,
            output.restarts,
            output.operator_applications,
        )
    };
    let residuals = buckling_residuals(k, negative_kg.as_ref(), solver, &values, &vectors)?;
    Ok(EigenResult {
        converged: lanczos_converged && values.len() == count,
        values,
        vectors,
        residuals,
        restarts,
        operator_applications,
        shift: 0.0,
        factorization: Some(solver.symbolic().estimate()),
        factor_kind: Some(FactorKind::Cholesky),
        sturm: None,
    })
}

/// Relative load-factor error bounds `η = ‖K φ − λ (−K_G) φ‖_{K⁻¹} / ‖φ‖_K` of buckling Ritz pairs.
///
/// `K` is SPD, so `(−K_G) φ = θ K φ` with `θ = 1/λ` is a definite pencil, and the residual
/// bound in the `K⁻¹` norm places an exact eigenvalue within `θ̃ (1 ± η)` (Parlett, *The
/// Symmetric Eigenvalue Problem*, Thm. 4.5.1 applied to `K^{-1/2}(−K_G)K^{-1/2}`). `η` is
/// therefore the relative load-factor error to first order, and it is invariant to the
/// scaling and conditioning of `K`. It is the quantity the `K`-inner-product Lanczos drives
/// below its tolerance; a Euclidean residual scaled by `‖K φ‖` is not, because `‖K φ‖` of a
/// smooth mode is small against `‖K‖ ‖φ‖`.
fn buckling_residuals(
    k: CsrRef<'_, f64>,
    negative_kg: CsrRef<'_, f64>,
    solver: &DirectSolver<f64>,
    values: &[f64],
    vectors: &[f64],
) -> SolverResult<Vec<f64>> {
    let n = k.nrows();
    let cols = values.len();
    if cols == 0 {
        return Ok(Vec::new());
    }
    let mut kx = vec![0.0; n * cols];
    let mut bx = vec![0.0; n * cols];
    blas::spmm(k, vectors, &mut kx, cols);
    blas::spmm(negative_kg, vectors, &mut bx, cols);
    let mut residual = kx.clone();
    for (c, &lambda) in values.iter().enumerate() {
        par::axpy(
            -lambda,
            &bx[c * n..(c + 1) * n],
            &mut residual[c * n..(c + 1) * n],
        );
    }
    let mut preconditioned = residual.clone();
    solver.solve_in_place(&mut preconditioned, cols)?;
    Ok((0..cols)
        .map(|c| {
            let range = c * n..(c + 1) * n;
            let energy = par::dot(&vectors[range.clone()], &kx[range.clone()]);
            let defect = par::dot(&residual[range.clone()], &preconditioned[range]).abs();
            if energy > 0.0 {
                (defect / energy).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect())
}

fn dense_buckling(
    k: CsrRef<'_, f64>,
    negative_kg: CsrRef<'_, f64>,
    count: usize,
) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    let n = k.nrows();
    // −K_G φ = θ K φ with θ = 1/λ.
    let (theta, x) = dense_generalized(negative_kg, k)?;
    let mut pairs: Vec<(f64, usize)> = theta
        .iter()
        .enumerate()
        .filter(|(_, &t)| t > 0.0)
        .map(|(i, &t)| (1.0 / t, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.truncate(count);
    let values: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let mut vectors = vec![0.0; n * values.len()];
    for (c, &(_, i)) in pairs.iter().enumerate() {
        vectors[c * n..(c + 1) * n].copy_from_slice(&x[i * n..(i + 1) * n]);
    }
    Ok((values, vectors))
}

/// Extreme eigenpairs of a real symmetric `A` without shift-invert (`eigs` with `'largestabs'`
/// or `'largestalgebraic'`): block Lanczos on `A` itself in the Euclidean inner product.
pub fn symmetric_extreme(
    a: CsrRef<'_, f64>,
    selection: Selection,
    options: &EigenOptions,
    control: SolveControl<'_>,
) -> SolverResult<EigenResult> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SolverError::Dimension(
            "eigenvalue operator must be square".into(),
        ));
    }
    let count = options.count.min(n);
    let p = options.block_size.max(1);
    let identity = CsrMatrix::<f64>::identity(n);
    if n <= 400 || n < count + 4 * p {
        let (values, vectors) = dense_generalized(a, identity.as_ref())?;
        let mut order: Vec<usize> = (0..n).collect();
        match selection {
            Selection::LargestAlgebraic => {
                order.sort_by(|&x, &y| values[y].partial_cmp(&values[x]).unwrap())
            }
            Selection::LargestMagnitude => {
                order.sort_by(|&x, &y| values[y].abs().partial_cmp(&values[x].abs()).unwrap())
            }
        }
        order.truncate(count);
        let picked: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        let mut picked_vectors = vec![0.0; n * count];
        for (c, &i) in order.iter().enumerate() {
            picked_vectors[c * n..(c + 1) * n].copy_from_slice(&vectors[i * n..(i + 1) * n]);
        }
        let residuals = true_residuals(a, identity.as_ref(), &picked, &picked_vectors);
        return Ok(EigenResult {
            converged: true,
            values: picked,
            vectors: picked_vectors,
            residuals,
            restarts: 0,
            operator_applications: 0,
            shift: 0.0,
            factorization: None,
            factor_kind: None,
            sturm: None,
        });
    }
    let config = LanczosConfig {
        n,
        nev: count,
        block: p,
        max_basis: basis_size(options, n),
        tolerance: options.tolerance,
        max_restarts: options.max_restarts,
        selection,
        seed: 0x7379_6d6d_6574_7269,
    };
    let mut apply_op = |x: &[f64], y: &mut [f64], cols: usize| -> SolverResult<()> {
        blas::spmm(a, x, y, cols);
        Ok(())
    };
    let apply_b = |x: &[f64], y: &mut [f64], _cols: usize| y.copy_from_slice(x);
    let output = thick_restart_block_lanczos(&config, &mut apply_op, &apply_b, control)?;
    let residuals = true_residuals(a, identity.as_ref(), &output.theta, &output.vectors);
    Ok(EigenResult {
        converged: output.converged >= count,
        values: output.theta,
        vectors: output.vectors,
        residuals,
        restarts: output.restarts,
        operator_applications: output.operator_applications,
        shift: 0.0,
        factorization: None,
        factor_kind: None,
        sturm: None,
    })
}

#[cfg(test)]
mod tests;
