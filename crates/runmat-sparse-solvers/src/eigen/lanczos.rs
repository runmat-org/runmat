//! Thick-restart block Lanczos for `Op = S⁻¹ A`, self-adjoint in the `B` inner product.
//!
//! The basis is kept fully `B`-orthonormal (block classical Gram–Schmidt, two passes), so the
//! projected matrix is formed explicitly and restarts keep the wanted Ritz vectors plus the
//! residual block (Krylov–Schur for the symmetric case). Blocks find clustered and repeated
//! eigenvalues (symmetric structures) reliably and turn the expensive part — applying `S⁻¹` —
//! into multi-right-hand-side triangular solves.

use super::blas::{gemm_nn, gemm_tn, ProductScale};
use crate::control::{Progress, ProgressPhase, SolveControl};
use crate::dense::symmetric_eigen;
use crate::error::{SolverError, SolverResult};
use crate::par;

/// Which Ritz values are wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    LargestAlgebraic,
    LargestMagnitude,
}

pub(crate) struct LanczosConfig {
    pub n: usize,
    pub nev: usize,
    pub block: usize,
    pub max_basis: usize,
    pub tolerance: f64,
    pub max_restarts: usize,
    pub selection: Selection,
    pub seed: u64,
}

pub(crate) struct LanczosOutput {
    /// Wanted Ritz values `θ`, best first.
    pub theta: Vec<f64>,
    /// Ritz vectors (`n × theta.len()`, column-major, `B`-orthonormal).
    pub vectors: Vec<f64>,
    pub converged: usize,
    pub restarts: usize,
    pub operator_applications: usize,
}

/// `apply_op(x, y, p)`: `y = Op x` for `p` columns; `apply_b(x, y, p)`: `y = B x`.
type ApplyOperator<'a> = dyn FnMut(&[f64], &mut [f64], usize) -> SolverResult<()> + 'a;
type ApplyMetric<'a> = dyn Fn(&[f64], &mut [f64], usize) + 'a;

pub(crate) fn thick_restart_block_lanczos(
    config: &LanczosConfig,
    apply_op: &mut ApplyOperator<'_>,
    apply_b: &ApplyMetric<'_>,
    control: SolveControl<'_>,
) -> SolverResult<LanczosOutput> {
    let n = config.n;
    let p = config.block.max(1);
    let m = config.max_basis.min(n);
    if config.nev == 0 {
        return Ok(LanczosOutput {
            theta: Vec::new(),
            vectors: Vec::new(),
            converged: 0,
            restarts: 0,
            operator_applications: 0,
        });
    }
    if m < config.nev + 2 * p {
        return Err(SolverError::Dimension(format!(
            "Lanczos basis {m} too small for {} eigenpairs with block {p}",
            config.nev
        )));
    }
    let mut v = par::try_zeros::<f64>(n * m)?;
    let mut t = vec![0.0f64; m * m];
    let mut applications = 0usize;
    let mut seed = config.seed;

    // Start block: Op applied to a deterministic random block (purifies components in the null
    // space of B), then B-orthonormalized.
    {
        let start = random_block(n, p, &mut seed);
        let mut w = vec![0.0; n * p];
        apply_op(&start, &mut w, p)?;
        applications += p;
        orthonormalize_block(&mut w, p, &v, 0, n, apply_b, &mut seed)?;
        v[..n * p].copy_from_slice(&w);
    }
    let mut k = 0usize;
    let mut restarts = 0usize;
    let mut w = vec![0.0; n * p];
    loop {
        control.check()?;
        // Expansion.
        while k + 2 * p <= m {
            control.check()?;
            apply_op(&v[k * n..(k + p) * n], &mut w, p)?;
            applications += p;
            let (h, r) = orthonormalize_block(&mut w, p, &v, k + p, n, apply_b, &mut seed)?;
            let rows = k + p;
            for j in 0..p {
                for i in 0..rows {
                    let value = h[j * rows + i];
                    t[(k + j) * m + i] = value;
                    t[i * m + k + j] = value;
                }
            }
            for j in 0..p {
                for i in 0..p {
                    // W_old = V_new R  ⇒  T[new_i, old_j] = R[i, j].
                    let value = r[j * p + i];
                    t[(k + j) * m + k + p + i] = value;
                    t[(k + p + i) * m + k + j] = value;
                }
            }
            v[(k + p) * n..(k + 2 * p) * n].copy_from_slice(&w);
            k += p;
        }
        // Rayleigh–Ritz on the complete part.
        let mut projected = vec![0.0; k * k];
        for j in 0..k {
            for i in 0..k {
                projected[j * k + i] = 0.5 * (t[j * m + i] + t[i * m + j]);
            }
        }
        let (theta, y) = symmetric_eigen(&projected, k)?;
        let mut order: Vec<usize> = (0..k).collect();
        match config.selection {
            Selection::LargestAlgebraic => {
                order.sort_by(|&a, &b| theta[b].partial_cmp(&theta[a]).unwrap())
            }
            Selection::LargestMagnitude => {
                order.sort_by(|&a, &b| theta[b].abs().partial_cmp(&theta[a].abs()).unwrap())
            }
        }
        // Residual estimates ‖C y‖ with C = T[k..k+p, 0..k].
        let residual = |index: usize| -> f64 {
            let mut sum = 0.0;
            for i in 0..p {
                let mut dot = 0.0;
                for j in 0..k {
                    dot += t[j * m + k + i] * y[index * k + j];
                }
                sum += dot * dot;
            }
            sum.sqrt()
        };
        let estimates: Vec<f64> = order.iter().map(|&i| residual(i)).collect();
        let converged = order
            .iter()
            .zip(&estimates)
            .take_while(|(&i, &e)| e <= config.tolerance * theta[i].abs().max(f64::MIN_POSITIVE))
            .count();
        control.report(Progress {
            phase: ProgressPhase::Eigen,
            iteration: restarts,
            residual: estimates.first().copied().unwrap_or(0.0),
            converged: converged.min(config.nev),
        });
        if converged >= config.nev || restarts >= config.max_restarts {
            let count = config.nev.min(k);
            let mut selected_y = vec![0.0; k * count];
            for (c, &index) in order.iter().take(count).enumerate() {
                selected_y[c * k..(c + 1) * k].copy_from_slice(&y[index * k..(index + 1) * k]);
            }
            let mut vectors = vec![0.0; n * count];
            gemm_nn(
                &mut vectors,
                count,
                ProductScale {
                    alpha: 1.0,
                    beta: 0.0,
                },
                &v[..k * n],
                k,
                &selected_y,
                n,
            );
            return Ok(LanczosOutput {
                theta: order.iter().take(count).map(|&i| theta[i]).collect(),
                vectors,
                converged: converged.min(count),
                restarts,
                operator_applications: applications,
            });
        }
        // Thick restart: keep the best Ritz vectors and the residual block.
        restarts += 1;
        let keep = (config.nev + p)
            .max(k / 2)
            .min(k - p)
            .max(config.nev.min(k - p));
        let mut kept_y = vec![0.0; k * keep];
        for (c, &index) in order.iter().take(keep).enumerate() {
            kept_y[c * k..(c + 1) * k].copy_from_slice(&y[index * k..(index + 1) * k]);
        }
        let mut new_basis = vec![0.0; n * keep];
        gemm_nn(
            &mut new_basis,
            keep,
            ProductScale {
                alpha: 1.0,
                beta: 0.0,
            },
            &v[..k * n],
            k,
            &kept_y,
            n,
        );
        // Coupling C·Y (p × keep).
        let mut coupling = vec![0.0; p * keep];
        for c in 0..keep {
            for i in 0..p {
                let mut dot = 0.0;
                for j in 0..k {
                    dot += t[j * m + k + i] * kept_y[c * k + j];
                }
                coupling[c * p + i] = dot;
            }
        }
        let residual_block = v[k * n..(k + p) * n].to_vec();
        v[..keep * n].copy_from_slice(&new_basis);
        v[keep * n..(keep + p) * n].copy_from_slice(&residual_block);
        t.iter_mut().for_each(|x| *x = 0.0);
        for (c, &index) in order.iter().take(keep).enumerate() {
            t[c * m + c] = theta[index];
        }
        for c in 0..keep {
            for i in 0..p {
                t[c * m + keep + i] = coupling[c * p + i];
                t[(keep + i) * m + c] = coupling[c * p + i];
            }
        }
        k = keep;
    }
}

fn random_block(n: usize, p: usize, seed: &mut u64) -> Vec<f64> {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    par::seeded_vector::<f64>(n * p, *seed)
}

/// B-orthogonalizes the block `w` (`n × p`) against `basis[:, 0..existing]` (two CGS passes) and
/// B-orthonormalizes it in place. Returns `H` (`existing × p`) and `R` (`p × p`), with
/// `w_in = basis H + w_out R`. Deflated columns are replaced by fresh random directions (their
/// `R` column is zero).
pub(crate) fn orthonormalize_block(
    w: &mut [f64],
    p: usize,
    basis: &[f64],
    existing: usize,
    n: usize,
    apply_b: &ApplyMetric<'_>,
    seed: &mut u64,
) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    let mut h = vec![0.0; existing * p];
    let mut r = vec![0.0; p * p];
    let mut bw = vec![0.0; n * p];
    let initial_norms: Vec<f64> = {
        apply_b(w, &mut bw, p);
        (0..p)
            .map(|c| {
                par::dot(&w[c * n..(c + 1) * n], &bw[c * n..(c + 1) * n])
                    .max(0.0)
                    .sqrt()
            })
            .collect()
    };
    for _pass in 0..2 {
        if existing == 0 {
            break;
        }
        apply_b(w, &mut bw, p);
        let coefficients = gemm_tn(&basis[..existing * n], existing, &bw, p, n);
        gemm_nn(
            w,
            p,
            ProductScale {
                alpha: -1.0,
                beta: 1.0,
            },
            &basis[..existing * n],
            existing,
            &coefficients,
            n,
        );
        for (target, value) in h.iter_mut().zip(&coefficients) {
            *target += value;
        }
    }
    let mut column_b = vec![0.0; n];
    for j in 0..p {
        let mut replaced = false;
        for attempt in 0..3 {
            for _pass in 0..2 {
                apply_b(&w[j * n..(j + 1) * n], &mut column_b, 1);
                for i in 0..j {
                    let c = par::dot(&w[i * n..(i + 1) * n], &column_b);
                    let (head, tail) = w.split_at_mut(j * n);
                    par::axpy(-c, &head[i * n..(i + 1) * n], &mut tail[..n]);
                    if !replaced {
                        r[j * p + i] += c;
                    }
                }
            }
            apply_b(&w[j * n..(j + 1) * n], &mut column_b, 1);
            let norm = par::dot(&w[j * n..(j + 1) * n], &column_b).max(0.0).sqrt();
            let reference = if replaced { 1.0 } else { initial_norms[j] };
            if norm > 1e-10 * reference.max(f64::MIN_POSITIVE) && norm.is_finite() {
                if !replaced {
                    r[j * p + j] = norm;
                }
                par::scale(1.0 / norm, &mut w[j * n..(j + 1) * n]);
                break;
            }
            if attempt == 2 {
                return Err(SolverError::Breakdown(
                    "cannot extend the B-orthonormal basis (B is numerically singular on the Krylov space)".into(),
                ));
            }
            // Deflate: new random direction, orthogonalized against the basis.
            replaced = true;
            for i in 0..p {
                if i != j {
                    // keep other columns
                }
            }
            let fresh = random_block(n, 1, seed);
            w[j * n..(j + 1) * n].copy_from_slice(&fresh);
            if existing > 0 {
                for _pass in 0..2 {
                    apply_b(&w[j * n..(j + 1) * n], &mut column_b, 1);
                    let coefficients = gemm_tn(&basis[..existing * n], existing, &column_b, 1, n);
                    gemm_nn(
                        &mut w[j * n..(j + 1) * n],
                        1,
                        ProductScale {
                            alpha: -1.0,
                            beta: 1.0,
                        },
                        &basis[..existing * n],
                        existing,
                        &coefficients,
                        n,
                    );
                }
            }
        }
    }
    Ok((h, r))
}
