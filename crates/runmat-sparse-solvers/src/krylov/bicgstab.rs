//! Right-preconditioned BiCGStab (van der Vorst) for general systems.

use super::{relative, split_work, tolerance, ApplyStatus, KrylovOps, KrylovOptions, KrylovReport, StopFlag};
use crate::error::SolverResult;
use crate::par;
use crate::scalar::Scalar;

pub fn bicgstab_workspace_len(n: usize, options: &KrylovOptions) -> usize {
    n * (7 + usize::from(options.return_best_iterate))
}

pub async fn bicgstab_async<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    work: &mut [T],
) -> SolverResult<KrylovReport> {
    let n = b.len();
    let mut parts = split_work(work, n, 7 + usize::from(options.return_best_iterate));
    let mut best = if options.return_best_iterate { parts.pop() } else { None };
    let mut it = parts.into_iter();
    let (r, rhat, p, v, t, phat, shat) = (
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
    );
    let mut counters = super::Counters {
        operator: 0,
        preconditioner: 0,
    };
    let rhs_norm = par::norm2(b);
    let tol = tolerance(options, rhs_norm);
    let mut history = Vec::new();
    if x.iter().all(|value| *value == T::zero()) {
        r.copy_from_slice(b);
    } else {
        ops.apply_operator(x, t).await?;
        counters.operator += 1;
        par::sub_into(b, t, r);
    }
    let mut residual = par::norm2(r);
    if options.record_history {
        history.push(residual);
    }
    let mut flag = StopFlag::MaxIterations;
    let mut iteration = 0usize;
    let (mut best_residual, mut best_iteration) = (residual, 0usize);
    if let Some(best) = best.as_deref_mut() {
        best.copy_from_slice(x);
    }
    if rhs_norm == 0.0 {
        x.iter_mut().for_each(|value| *value = T::zero());
        residual = 0.0;
        flag = StopFlag::Converged;
    } else if residual <= tol {
        flag = StopFlag::Converged;
    } else {
        rhat.copy_from_slice(r);
        let (mut rho_old, mut alpha, mut omega) = (T::one(), T::one(), T::one());
        p.iter_mut().for_each(|value| *value = T::zero());
        v.iter_mut().for_each(|value| *value = T::zero());
        let mut stagnant = 0usize;
        while iteration < options.max_iterations {
            iteration += 1;
            ops.checkpoint(iteration, relative(residual, rhs_norm)).await?;
            let rho = par::dot(rhat, r);
            if rho.abs() == 0.0 || !rho.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            if iteration == 1 {
                p.copy_from_slice(r);
            } else {
                let beta = (rho / rho_old) * (alpha / omega);
                // p = r + beta (p - omega v)
                par::axpy(-omega, v, p);
                par::xpby(r, beta, p);
            }
            if ops.has_preconditioner() {
                counters.preconditioner += 1;
                if ops.apply_preconditioner(p, phat).await? == ApplyStatus::NumericalFailure {
                    flag = StopFlag::PreconditionerFailure;
                    break;
                }
            } else {
                phat.copy_from_slice(p);
            }
            ops.apply_operator(phat, v).await?;
            counters.operator += 1;
            let denominator = par::dot(rhat, v);
            if denominator.abs() == 0.0 || !denominator.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            alpha = rho / denominator;
            // s = r - alpha v (stored in r)
            par::axpy(-alpha, v, r);
            let s_norm = par::norm2(r);
            if s_norm <= tol {
                par::axpy(alpha, phat, x);
                residual = s_norm;
                if options.record_history {
                    history.push(residual);
                }
                flag = StopFlag::Converged;
                break;
            }
            if ops.has_preconditioner() {
                counters.preconditioner += 1;
                if ops.apply_preconditioner(r, shat).await? == ApplyStatus::NumericalFailure {
                    flag = StopFlag::PreconditionerFailure;
                    break;
                }
            } else {
                shat.copy_from_slice(r);
            }
            ops.apply_operator(shat, t).await?;
            counters.operator += 1;
            let tt = par::dot(t, t).re();
            if tt == 0.0 || !tt.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            omega = par::dot(t, r) / T::from_f64(tt);
            let step_norm = alpha.abs() * par::norm2(phat) + omega.abs() * par::norm2(shat);
            if step_norm <= T::EPSILON * par::norm2(x) {
                stagnant += 1;
            } else {
                stagnant = 0;
            }
            par::axpy(alpha, phat, x);
            par::axpy(omega, shat, x);
            par::axpy(-omega, t, r);
            residual = par::norm2(r);
            if options.record_history {
                history.push(residual);
            }
            if residual < best_residual {
                best_residual = residual;
                best_iteration = iteration;
                if let Some(best) = best.as_deref_mut() {
                    best.copy_from_slice(x);
                }
            }
            if residual <= tol {
                flag = StopFlag::Converged;
                break;
            }
            if omega.abs() == 0.0 {
                flag = StopFlag::Breakdown;
                break;
            }
            if stagnant >= options.max_stagnation_steps {
                flag = StopFlag::Stagnation;
                break;
            }
            rho_old = rho;
        }
    }
    // True residual of the returned iterate.
    if flag != StopFlag::Converged {
        if let Some(best_x) = best.as_deref() {
            x.copy_from_slice(best_x);
            iteration = best_iteration;
        }
    }
    if rhs_norm > 0.0 {
        ops.apply_operator(x, t).await?;
        counters.operator += 1;
        par::sub_into(b, t, r);
        residual = par::norm2(r);
        if flag == StopFlag::Converged && residual > tol * 10.0 {
            // Recursive residual drifted from the true residual.
            flag = StopFlag::Stagnation;
        }
    }
    Ok(KrylovReport {
        flag,
        iterations: iteration,
        relative_residual: relative(residual, rhs_norm),
        residual_norm: residual,
        rhs_norm,
        residual_history: history,
        restart_position: None,
        operator_applications: counters.operator,
        preconditioner_applications: counters.preconditioner,
        spectrum: None,
    })
}
