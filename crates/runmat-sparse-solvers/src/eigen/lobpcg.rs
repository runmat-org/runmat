//! LOBPCG (Knyazev) for the lowest eigenpairs of `K x = λ M x` with an SPD preconditioner
//! `T ≈ K⁻¹` (typically AMG). Each step M-orthogonalizes the search directions `[W, P]` against
//! the iterate `X` and M-orthonormalizes them by an eigen-decomposition of their Gram matrix
//! (dropping numerically dependent directions), as Hetmaniuk and Lehoucq (2006) do. On an
//! M-orthonormal basis the Rayleigh–Ritz Gram matrices stay well conditioned while the residuals
//! shrink, so the iteration reaches tight residual tolerances instead of stagnating where `W`
//! falls into the span of `X` and `P`.

use super::blas::{gemm_nn, gemm_tn, spmm, ProductScale};
use super::{eigenpair_backward_error, true_residuals, EigenResult, ResidualMeasure};
use crate::control::{Progress, ProgressPhase, SolveControl};
use crate::csr::CsrRef;
use crate::dense::symmetric_eigen;
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::par;

#[derive(Debug, Clone, PartialEq)]
pub struct LobpcgOptions {
    pub count: usize,
    /// Extra block vectors that accelerate convergence of the wanted ones.
    pub guard: usize,
    /// Stopping tolerance of each wanted pair, in `measure`.
    pub tolerance: f64,
    /// The residual measure the stopping test applies `tolerance` to.
    pub measure: ResidualMeasure,
    pub max_iterations: usize,
}

impl Default for LobpcgOptions {
    fn default() -> Self {
        Self {
            count: 10,
            guard: 4,
            tolerance: 1e-8,
            measure: ResidualMeasure::RelativeResidual,
            max_iterations: 500,
        }
    }
}

/// Block width LOBPCG iterates with for `options` on an `n`-dimensional pencil.
pub fn lobpcg_block_size(options: &LobpcgOptions, n: usize) -> usize {
    (options.count + options.guard)
        .min(n / 3)
        .max(options.count.min(n))
}

/// Upper bound on the `f64` values [`lobpcg`] holds at once: the iterate, its operator and
/// metric images, the residual, the preconditioned residual and the previous direction (one
/// block each), the Rayleigh–Ritz basis with its two images (three blocks each), the next
/// iterate and direction, the projected Gram matrices, and the returned pairs with their
/// residual scratch. Returns `None` on arithmetic overflow.
pub fn lobpcg_working_value_count(options: &LobpcgOptions, n: usize) -> Option<usize> {
    let block = lobpcg_block_size(options, n);
    let basis = block.checked_mul(3)?;
    let blocks = n.checked_mul(block)?.checked_mul(17)?;
    let projected = basis.checked_mul(basis)?.checked_mul(6)?;
    let returned = n.checked_mul(options.count.min(n).checked_add(2)?)?;
    blocks.checked_add(projected)?.checked_add(returned)
}

/// Solves the projected problem `Sᵀ K S y = θ Sᵀ M S y` after removing dependent directions.
fn rayleigh_ritz(gk: &[f64], gm: &[f64], s: usize) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    // Scale to unit diagonal of Gm, eigen-decompose Gm, keep well-conditioned directions.
    let d: Vec<f64> = (0..s)
        .map(|i| gm[i * s + i].max(f64::MIN_POSITIVE).sqrt().recip())
        .collect();
    let scaled_m: Vec<f64> = (0..s * s).map(|e| gm[e] * d[e / s] * d[e % s]).collect();
    let (mu, u) = symmetric_eigen(&scaled_m, s)?;
    let top = mu.iter().fold(0.0f64, |a, &b| a.max(b));
    let keep: Vec<usize> = (0..s).filter(|&i| mu[i] > 1e-12 * top).collect();
    let r = keep.len();
    // Basis transform Z = D U_keep diag(mu^-1/2) (s × r).
    let mut z = vec![0.0; s * r];
    for (c, &i) in keep.iter().enumerate() {
        let scale = mu[i].sqrt().recip();
        for row in 0..s {
            z[c * s + row] = d[row] * u[i * s + row] * scale;
        }
    }
    // C = Zᵀ Gk Z.
    let mut gz = vec![0.0; s * r];
    for c in 0..r {
        for row in 0..s {
            let mut sum = 0.0;
            for k in 0..s {
                sum += gk[k * s + row] * z[c * s + k];
            }
            gz[c * s + row] = sum;
        }
    }
    let mut cmat = vec![0.0; r * r];
    for j in 0..r {
        for i in 0..r {
            let mut sum = 0.0;
            for k in 0..s {
                sum += z[i * s + k] * gz[j * s + k];
            }
            cmat[j * r + i] = sum;
        }
    }
    for j in 0..r {
        for i in 0..j {
            let avg = 0.5 * (cmat[j * r + i] + cmat[i * r + j]);
            cmat[j * r + i] = avg;
            cmat[i * r + j] = avg;
        }
    }
    let (theta, w) = symmetric_eigen(&cmat, r)?;
    // y = Z w (s × r).
    let mut y = vec![0.0; s * r];
    for c in 0..r {
        for row in 0..s {
            let mut sum = 0.0;
            for k in 0..r {
                sum += z[k * s + row] * w[c * r + k];
            }
            y[c * s + row] = sum;
        }
    }
    Ok((theta, y))
}

/// M-orthogonalizes the `c = v.len() / n` columns of `v` against the M-orthonormal block `x`
/// (with `mx = M x`) and M-orthonormalizes what remains, dropping numerically dependent
/// directions. Two passes recover the orthogonality the first loses to cancellation. Returns the
/// basis and its image under `M`.
fn m_orthonormal_complement(
    m: CsrRef<'_, f64>,
    x: &[f64],
    mx: &[f64],
    block: usize,
    mut v: Vec<f64>,
    n: usize,
) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    let mut mv = Vec::new();
    for _ in 0..2 {
        let c = v.len() / n;
        if c == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        // V ← V − X (M X)ᵀ V.
        let coefficients = gemm_tn(mx, block, &v, c, n);
        gemm_nn(
            &mut v,
            c,
            ProductScale {
                alpha: -1.0,
                beta: 1.0,
            },
            x,
            block,
            &coefficients,
            n,
        );
        mv = vec![0.0; n * c];
        spmm(m, &v, &mut mv, c);
        let gram = gemm_tn(&v, c, &mv, c, n);
        let z = orthonormalizing_transform(&gram, c)?;
        let r = z.len() / c.max(1);
        let mut orthonormal = vec![0.0; n * r];
        gemm_nn(
            &mut orthonormal,
            r,
            ProductScale {
                alpha: 1.0,
                beta: 0.0,
            },
            &v,
            c,
            &z,
            n,
        );
        let mut m_orthonormal = vec![0.0; n * r];
        gemm_nn(
            &mut m_orthonormal,
            r,
            ProductScale {
                alpha: 1.0,
                beta: 0.0,
            },
            &mv,
            c,
            &z,
            n,
        );
        v = orthonormal;
        mv = m_orthonormal;
    }
    Ok((v, mv))
}

/// `Z = D U_keep diag(μ^-1/2)` for the Gram matrix `G` (c × c): `Zᵀ G Z = I` on the directions
/// whose unit-diagonal-scaled eigenvalue exceeds a rounding-level fraction of the largest.
fn orthonormalizing_transform(gram: &[f64], c: usize) -> SolverResult<Vec<f64>> {
    let d: Vec<f64> = (0..c)
        .map(|i| gram[i * c + i].max(f64::MIN_POSITIVE).sqrt().recip())
        .collect();
    let scaled: Vec<f64> = (0..c * c).map(|e| gram[e] * d[e / c] * d[e % c]).collect();
    let (mu, u) = symmetric_eigen(&scaled, c)?;
    let top = mu.iter().fold(0.0f64, |a, &b| a.max(b));
    let keep: Vec<usize> = (0..c)
        .filter(|&i| mu[i] > 1e-13 * top && mu[i] > 0.0)
        .collect();
    let mut z = vec![0.0; c * keep.len()];
    for (column, &i) in keep.iter().enumerate() {
        let scale = mu[i].sqrt().recip();
        for row in 0..c {
            z[column * c + row] = d[row] * u[i * c + row] * scale;
        }
    }
    Ok(z)
}

pub fn lobpcg(
    k: CsrRef<'_, f64>,
    m: CsrRef<'_, f64>,
    preconditioner: Option<&dyn Preconditioner<f64>>,
    options: &LobpcgOptions,
    control: SolveControl<'_>,
) -> SolverResult<EigenResult> {
    let n = k.nrows();
    let block = lobpcg_block_size(options, n);
    if block == 0 {
        return Err(SolverError::Dimension("empty eigenproblem".into()));
    }
    let mut x = par::seeded_vector::<f64>(n * block, 0x6c6f_6270_6367);
    let mut p_block: Option<Vec<f64>> = None;
    let mut theta = vec![0.0; block];
    let mut converged_count;
    let mut iterations = 0;
    let mut applications = 0;
    let mut kx = vec![0.0; n * block];
    let mut mx = vec![0.0; n * block];
    let pencil_norms = match options.measure {
        ResidualMeasure::BackwardError => Some((k.norm_inf(), m.norm_inf())),
        ResidualMeasure::RelativeResidual => None,
    };
    // Initial Rayleigh–Ritz on X.
    {
        spmm(k, &x, &mut kx, block);
        spmm(m, &x, &mut mx, block);
        let gk = gemm_tn(&x, block, &kx, block, n);
        let gm = gemm_tn(&x, block, &mx, block, n);
        let (t, y) = rayleigh_ritz(&gk, &gm, block)?;
        let r = t.len();
        if r < block {
            return Err(SolverError::Breakdown(
                "dependent LOBPCG start block".into(),
            ));
        }
        let mut xn = vec![0.0; n * block];
        gemm_nn(
            &mut xn,
            block,
            ProductScale {
                alpha: 1.0,
                beta: 0.0,
            },
            &x,
            block,
            &y[..block * block],
            n,
        );
        x = xn;
        theta.copy_from_slice(&t[..block]);
    }
    loop {
        control.check()?;
        iterations += 1;
        spmm(k, &x, &mut kx, block);
        spmm(m, &x, &mut mx, block);
        // Residuals R = K X − M X Θ.
        let mut residual = kx.clone();
        let mut norms = vec![0.0; block];
        for c in 0..block {
            let (rc, mc) = (&mut residual[c * n..(c + 1) * n], &mx[c * n..(c + 1) * n]);
            par::axpy(-theta[c], mc, rc);
            norms[c] = match pencil_norms {
                Some((k_norm, m_norm)) => eigenpair_backward_error(
                    par::norm_inf(rc),
                    theta[c],
                    k_norm,
                    m_norm,
                    par::norm_inf(&x[c * n..(c + 1) * n]),
                ),
                None => {
                    let kn = par::norm2(&kx[c * n..(c + 1) * n]);
                    let mn = par::norm2(mc);
                    par::norm2(rc) / (kn + theta[c].abs() * mn).max(f64::MIN_POSITIVE)
                }
            };
        }
        converged_count = norms
            .iter()
            .take(options.count)
            .take_while(|&&e| e <= options.tolerance)
            .count();
        control.report(Progress {
            phase: ProgressPhase::Eigen,
            iteration: iterations,
            residual: norms
                .iter()
                .take(options.count)
                .fold(0.0f64, |a, &b| a.max(b)),
            converged: converged_count,
        });
        if converged_count >= options.count || iterations >= options.max_iterations {
            break;
        }
        // Active (unconverged) columns get preconditioned residuals.
        let active: Vec<usize> = (0..block)
            .filter(|&c| norms[c] > options.tolerance)
            .collect();
        let a = active.len();
        let mut w = vec![0.0; n * a];
        for (j, &c) in active.iter().enumerate() {
            let rc = &residual[c * n..(c + 1) * n];
            match preconditioner {
                Some(t) => t.apply(rc, &mut w[j * n..(j + 1) * n])?,
                None => w[j * n..(j + 1) * n].copy_from_slice(rc),
            }
            applications += 1;
        }
        // Basis S = [X, Q] with Q an M-orthonormal basis of [W, P] M-orthogonal to X.
        let mut directions = w;
        if let Some(p) = p_block.take() {
            directions.extend_from_slice(&p);
        }
        let mut basis = Vec::with_capacity(n * (block + directions.len() / n));
        basis.extend_from_slice(&x);
        let (directions, m_directions) =
            m_orthonormal_complement(m, &x, &mx, block, directions, n)?;
        basis.extend_from_slice(&directions);
        drop(directions);
        let s = basis.len() / n;
        let mut kb = vec![0.0; n * s];
        let mut mb = vec![0.0; n * s];
        kb[..n * block].copy_from_slice(&kx);
        mb[..n * block].copy_from_slice(&mx);
        spmm(k, &basis[n * block..], &mut kb[n * block..], s - block);
        mb[n * block..].copy_from_slice(&m_directions);
        drop(m_directions);
        let gk = gemm_tn(&basis, s, &kb, s, n);
        let gm = gemm_tn(&basis, s, &mb, s, n);
        let (t, y) = rayleigh_ritz(&gk, &gm, s)?;
        if t.len() < block {
            // Lost rank: restart P.
            p_block = None;
            continue;
        }
        let r = t.len();
        let _ = r;
        // X_new = S Y[:, :block]; P_new = (S without X) Y_rest-part.
        let ycols = &y[..s * block];
        let mut x_new = vec![0.0; n * block];
        gemm_nn(
            &mut x_new,
            block,
            ProductScale {
                alpha: 1.0,
                beta: 0.0,
            },
            &basis,
            s,
            ycols,
            n,
        );
        let mut y_tail = vec![0.0; (s - block) * block];
        for c in 0..block {
            for row in block..s {
                y_tail[c * (s - block) + row - block] = ycols[c * s + row];
            }
        }
        let mut p_new = vec![0.0; n * block];
        gemm_nn(
            &mut p_new,
            block,
            ProductScale {
                alpha: 1.0,
                beta: 0.0,
            },
            &basis[n * block..],
            s - block,
            &y_tail,
            n,
        );
        x = x_new;
        p_block = Some(p_new);
        theta.copy_from_slice(&t[..block]);
    }
    let count = options.count.min(block);
    let values = theta[..count].to_vec();
    let vectors = x[..n * count].to_vec();
    let (residuals, backward_errors) = true_residuals(k, m, &values, &vectors);
    Ok(EigenResult {
        converged: converged_count >= options.count,
        values,
        vectors,
        residuals,
        backward_errors,
        restarts: iterations,
        operator_applications: applications,
        shift: 0.0,
        factorization: None,
        factor_kind: None,
        sturm: None,
    })
}
