//! Preconditioned conjugate gradients.

use super::{relative, split_work, tolerance, ApplyStatus, KrylovOps, KrylovOptions, KrylovReport, StopFlag};
use crate::error::SolverResult;
use crate::par;
use crate::scalar::Scalar;

/// Workspace entries required by [`pcg_async`].
pub fn pcg_workspace_len(n: usize, options: &KrylovOptions) -> usize {
    n * (4 + usize::from(options.return_best_iterate))
}

/// PCG for Hermitian positive definite systems. `x` carries the initial guess in and the returned
/// iterate out; `work` must hold [`pcg_workspace_len`] entries.
pub async fn pcg_async<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    work: &mut [T],
) -> SolverResult<KrylovReport> {
    let n = b.len();
    let mut parts = split_work(work, n, 4 + usize::from(options.return_best_iterate));
    let best = if options.return_best_iterate { parts.pop() } else { None };
    let mut parts = parts.into_iter();
    let (r, z, p, q) = (
        parts.next().unwrap(),
        parts.next().unwrap(),
        parts.next().unwrap(),
        parts.next().unwrap(),
    );
    let mut best = best;
    let mut counters = super::Counters {
        operator: 0,
        preconditioner: 0,
    };
    let rhs_norm = par::norm2(b);
    let tol = tolerance(options, rhs_norm);
    let mut history = Vec::new();

    // r = b - A x0
    let x_is_zero = x.iter().all(|v| *v == T::zero());
    if x_is_zero {
        r.copy_from_slice(b);
    } else {
        ops.apply_operator(x, q).await?;
        counters.operator += 1;
        par::sub_into(b, q, r);
    }
    let mut residual = par::norm2(r);
    if options.record_history {
        history.push(residual);
    }
    let finish = |flag, iterations, residual: f64, history: Vec<f64>, counters: &super::Counters| KrylovReport {
        flag,
        iterations,
        relative_residual: relative(residual, rhs_norm),
        residual_norm: residual,
        rhs_norm,
        residual_history: history,
        restart_position: None,
        operator_applications: counters.operator,
        preconditioner_applications: counters.preconditioner,
    };
    if residual <= tol || rhs_norm == 0.0 && residual == 0.0 {
        if rhs_norm == 0.0 {
            x.iter_mut().for_each(|v| *v = T::zero());
            return Ok(finish(StopFlag::Converged, 0, 0.0, history, &counters));
        }
        return Ok(finish(StopFlag::Converged, 0, residual, history, &counters));
    }
    if let Some(best) = best.as_deref_mut() {
        best.copy_from_slice(x);
    }
    let (mut best_residual, mut best_iteration) = (residual, 0usize);
    let mut rho_previous = 0.0f64;
    let mut stagnant = 0usize;
    let mut verification_rounds = 0usize;
    let mut flag = StopFlag::MaxIterations;
    let mut iteration = 0usize;

    while iteration < options.max_iterations {
        iteration += 1;
        ops.checkpoint(iteration, relative(residual, rhs_norm)).await?;
        if ops.has_preconditioner() {
            counters.preconditioner += 1;
            if ops.apply_preconditioner(r, z).await? == ApplyStatus::NumericalFailure
                || z.iter().any(|v| !v.is_finite())
            {
                flag = StopFlag::PreconditionerFailure;
                break;
            }
        } else {
            z.copy_from_slice(r);
        }
        let rho_value = par::dot(r, z);
        let rho = rho_value.re();
        if !rho.is_finite() || rho <= 0.0 {
            flag = if ops.has_preconditioner() && rho < 0.0 {
                StopFlag::PreconditionerFailure
            } else {
                StopFlag::Breakdown
            };
            break;
        }
        if iteration == 1 {
            p.copy_from_slice(z);
        } else {
            let beta = rho / rho_previous;
            if !beta.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            par::xpby(z, T::from_f64(beta), p);
        }
        ops.apply_operator(p, q).await?;
        counters.operator += 1;
        let curvature = par::dot(p, q).re();
        if !curvature.is_finite() || curvature <= 0.0 {
            flag = StopFlag::Breakdown;
            break;
        }
        let alpha = rho / curvature;
        // Stagnation: the update no longer changes x at working precision.
        let step = alpha.abs() * par::norm2(p);
        let x_norm = par::norm2(x);
        if step <= T::EPSILON * x_norm {
            stagnant += 1;
        } else {
            stagnant = 0;
        }
        par::axpy(T::from_f64(alpha), p, x);
        par::axpy(T::from_f64(-alpha), q, r);
        rho_previous = rho;
        residual = par::norm2(r);
        if options.record_history {
            history.push(residual);
        }
        if !residual.is_finite() {
            flag = StopFlag::Breakdown;
            break;
        }
        if residual <= tol {
            // Verify with the true residual; replace the recursive one if they disagree.
            ops.apply_operator(x, q).await?;
            counters.operator += 1;
            par::sub_into(b, q, r);
            residual = par::norm2(r);
            if residual <= tol {
                flag = StopFlag::Converged;
                break;
            }
            verification_rounds += 1;
            if verification_rounds > options.max_verification_rounds {
                flag = StopFlag::Stagnation;
                break;
            }
            // Restart the recurrence from the true residual.
            iteration_restart(&mut rho_previous);
            stagnant = 0;
            if residual < best_residual {
                best_residual = residual;
                best_iteration = iteration;
                if let Some(best) = best.as_deref_mut() {
                    best.copy_from_slice(x);
                }
            }
            // Next iteration starts a fresh direction.
            p.iter_mut().for_each(|v| *v = T::zero());
            continue;
        }
        if residual < best_residual {
            best_residual = residual;
            best_iteration = iteration;
            if let Some(best) = best.as_deref_mut() {
                best.copy_from_slice(x);
            }
        }
        if stagnant >= options.max_stagnation_steps {
            flag = StopFlag::Stagnation;
            break;
        }
    }

    if flag == StopFlag::Converged {
        return Ok(finish(flag, iteration, residual, history, &counters));
    }
    // Failure: return the best iterate (true residual) when requested, else the last iterate.
    let returned_iteration = if let Some(best) = best.as_deref() {
        x.copy_from_slice(best);
        best_iteration
    } else {
        iteration
    };
    ops.apply_operator(x, q).await?;
    counters.operator += 1;
    par::sub_into(b, q, r);
    let true_residual = par::norm2(r);
    Ok(finish(flag, returned_iteration, true_residual, history, &counters))
}

/// After a true-residual replacement the CG recurrence restarts (β = 0 on the next step).
fn iteration_restart(rho_previous: &mut f64) {
    *rho_previous = f64::INFINITY;
}
