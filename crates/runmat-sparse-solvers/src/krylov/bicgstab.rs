//! Right-preconditioned BiCGStab (van der Vorst) for general systems.

use super::legacy_usage as measured;
use super::usage::ScalarWork;
use super::{
    relative, split_work, tolerance, ApplyStatus, KrylovOps, KrylovOptions, KrylovReport, StopFlag,
};
use crate::error::SolverResult;
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
    let mut best = if options.return_best_iterate {
        parts.pop()
    } else {
        None
    };
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
    let rhs_norm = measured::norm(ops, b).await?;
    let tol =
        measured::scalar::<T, _, _>(ops, ScalarWork::NormScale, || tolerance(options, rhs_norm))
            .await?;
    let mut history = Vec::new();
    if x.iter().all(|value| *value == T::zero()) {
        r.copy_from_slice(b);
    } else {
        ops.apply_operator(x, t).await?;
        counters.operator += 1;
        measured::sub(ops, b, t, r).await?;
    }
    let mut residual = measured::norm(ops, r).await?;
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
            let relative_residual = measured::scalar::<T, _, _>(ops, ScalarWork::Ratio, || {
                relative(residual, rhs_norm)
            })
            .await?;
            ops.checkpoint(iteration, relative_residual).await?;
            let rho = measured::dot(ops, rhat, r).await?;
            if measured::magnitude(ops, rho).await? == 0.0 || !rho.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            if iteration == 1 {
                p.copy_from_slice(r);
            } else {
                let beta = measured::scalar::<T, _, _>(
                    ops,
                    ScalarWork::BiCgBeta {
                        domain: measured::arithmetic_domain::<T>(),
                    },
                    || (rho / rho_old) * (alpha / omega),
                )
                .await?;
                // p = r + beta (p - omega v)
                measured::axpy(ops, -omega, v, p).await?;
                measured::xpby(ops, r, beta, p).await?;
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
            let denominator = measured::dot(ops, rhat, v).await?;
            if measured::magnitude(ops, denominator).await? == 0.0 || !denominator.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            alpha = measured::scalar::<T, _, _>(
                ops,
                ScalarWork::BiCgRatio {
                    domain: measured::arithmetic_domain::<T>(),
                },
                || rho / denominator,
            )
            .await?;
            // s = r - alpha v (stored in r)
            measured::axpy(ops, -alpha, v, r).await?;
            let s_norm = measured::norm(ops, r).await?;
            if s_norm <= tol {
                measured::axpy(ops, alpha, phat, x).await?;
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
            let tt = measured::dot(ops, t, t).await?.re();
            if tt == 0.0 || !tt.is_finite() {
                flag = StopFlag::Breakdown;
                break;
            }
            let omega_numerator = measured::dot(ops, t, r).await?;
            omega = measured::scalar::<T, _, _>(
                ops,
                ScalarWork::BiCgDivideReal {
                    domain: measured::arithmetic_domain::<T>(),
                },
                || omega_numerator / T::from_f64(tt),
            )
            .await?;
            let phat_norm = measured::norm(ops, phat).await?;
            let shat_norm = measured::norm(ops, shat).await?;
            let x_norm = measured::norm(ops, x).await?;
            let complex_magnitudes = u8::from(T::IS_COMPLEX && alpha.im() != 0.0)
                + u8::from(T::IS_COMPLEX && omega.im() != 0.0);
            let (step_norm, threshold) = measured::scalar::<T, _, _>(
                ops,
                ScalarWork::BiCgStepNorm { complex_magnitudes },
                || {
                    let alpha_magnitude = if alpha.im() == 0.0 {
                        alpha.re().abs()
                    } else {
                        alpha.abs()
                    };
                    let omega_magnitude = if omega.im() == 0.0 {
                        omega.re().abs()
                    } else {
                        omega.abs()
                    };
                    (
                        alpha_magnitude * phat_norm + omega_magnitude * shat_norm,
                        T::EPSILON * x_norm,
                    )
                },
            )
            .await?;
            if step_norm <= threshold {
                stagnant += 1;
            } else {
                stagnant = 0;
            }
            measured::axpy(ops, alpha, phat, x).await?;
            measured::axpy(ops, omega, shat, x).await?;
            measured::axpy(ops, -omega, t, r).await?;
            residual = measured::norm(ops, r).await?;
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
            if measured::magnitude(ops, omega).await? == 0.0 {
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
        measured::sub(ops, b, t, r).await?;
        residual = measured::norm(ops, r).await?;
        let drift_limit =
            measured::scalar::<T, _, _>(ops, ScalarWork::NormScale, || tol * 10.0).await?;
        if flag == StopFlag::Converged && residual > drift_limit {
            // Recursive residual drifted from the true residual.
            flag = StopFlag::Stagnation;
        }
    }
    Ok(KrylovReport {
        flag,
        iterations: iteration,
        relative_residual: measured::scalar::<T, _, _>(
            ops,
            if rhs_norm > 0.0 {
                ScalarWork::Ratio
            } else {
                ScalarWork::Zero
            },
            || relative(residual, rhs_norm),
        )
        .await?,
        residual_norm: residual,
        rhs_norm,
        residual_history: history,
        restart_position: None,
        operator_applications: counters.operator,
        preconditioner_applications: counters.preconditioner,
        spectrum: None,
    })
}
