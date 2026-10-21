//! MINRES (Paige–Saunders) for Hermitian, possibly indefinite systems with an HPD preconditioner.

use super::{relative, split_work, tolerance, ApplyStatus, KrylovOps, KrylovOptions, KrylovReport, StopFlag};
use crate::error::SolverResult;
use crate::par;
use crate::scalar::Scalar;

pub fn minres_workspace_len(n: usize, options: &KrylovOptions) -> usize {
    n * (7 + usize::from(options.return_best_iterate))
}

pub async fn minres_async<T: Scalar, O: KrylovOps<T>>(
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
    let (r1, r2, y, v, w, w1, w2) = (
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
    let mut iteration = 0usize;
    let mut verification_rounds = 0usize;
    let mut flag = StopFlag::MaxIterations;
    let (mut best_residual, mut best_iteration) = (f64::INFINITY, 0usize);
    let mut true_residual;

    'restart: loop {
        // r1 = b - A x
        if x.iter().all(|value| *value == T::zero()) {
            r1.copy_from_slice(b);
        } else {
            ops.apply_operator(x, y).await?;
            counters.operator += 1;
            par::sub_into(b, y, r1);
        }
        true_residual = par::norm2(r1);
        if options.record_history && history.is_empty() {
            history.push(true_residual);
        }
        if true_residual < best_residual {
            best_residual = true_residual;
            best_iteration = iteration;
            if let Some(best) = best.as_deref_mut() {
                best.copy_from_slice(x);
            }
        }
        if true_residual <= tol || rhs_norm == 0.0 {
            if rhs_norm == 0.0 {
                x.iter_mut().for_each(|value| *value = T::zero());
                true_residual = 0.0;
            }
            flag = StopFlag::Converged;
            break 'restart;
        }
        if ops.has_preconditioner() {
            counters.preconditioner += 1;
            if ops.apply_preconditioner(r1, y).await? == ApplyStatus::NumericalFailure {
                flag = StopFlag::PreconditionerFailure;
                break 'restart;
            }
        } else {
            y.copy_from_slice(r1);
        }
        let beta1_sq = par::dot(r1, y).re();
        if !beta1_sq.is_finite() || beta1_sq < 0.0 {
            flag = StopFlag::PreconditionerFailure;
            break 'restart;
        }
        let beta1 = beta1_sq.sqrt();
        if beta1 == 0.0 {
            flag = StopFlag::Converged;
            break 'restart;
        }
        // The preconditioned residual estimate and the true residual are related by
        // ‖r‖ ≈ (‖r₀‖ / β₁) · φ̄; scale the stopping test accordingly.
        let scale = true_residual / beta1;
        r2.copy_from_slice(r1);
        w.iter_mut().for_each(|value| *value = T::zero());
        w2.iter_mut().for_each(|value| *value = T::zero());
        let (mut oldb, mut beta, mut dbar, mut epsln) = (0.0f64, beta1, 0.0f64, 0.0f64);
        let mut phibar = beta1;
        let (mut cs, mut sn) = (-1.0f64, 0.0f64);
        let mut first = true;
        while iteration < options.max_iterations {
            iteration += 1;
            ops.checkpoint(iteration, relative(phibar * scale, rhs_norm)).await?;
            let s = 1.0 / beta;
            for i in 0..n {
                v[i] = y[i].scale(s);
            }
            ops.apply_operator(v, y).await?;
            counters.operator += 1;
            if !first {
                par::axpy(T::from_f64(-beta / oldb), r1, y);
            }
            first = false;
            let alfa = par::dot(v, y).re();
            par::axpy(T::from_f64(-alfa / beta), r2, y);
            r1.copy_from_slice(r2);
            r2.copy_from_slice(y);
            if ops.has_preconditioner() {
                counters.preconditioner += 1;
                if ops.apply_preconditioner(r2, y).await? == ApplyStatus::NumericalFailure {
                    flag = StopFlag::PreconditionerFailure;
                    break 'restart;
                }
            } else {
                y.copy_from_slice(r2);
            }
            oldb = beta;
            let beta_sq = par::dot(r2, y).re();
            if !beta_sq.is_finite() || beta_sq < 0.0 {
                flag = StopFlag::PreconditionerFailure;
                break 'restart;
            }
            beta = beta_sq.sqrt();
            let oldeps = epsln;
            let delta = cs * dbar + sn * alfa;
            let gbar = sn * dbar - cs * alfa;
            epsln = sn * beta;
            dbar = -cs * beta;
            let gamma = gbar.hypot(beta).max(f64::EPSILON);
            cs = gbar / gamma;
            sn = beta / gamma;
            let phi = cs * phibar;
            phibar *= sn;
            let denom = 1.0 / gamma;
            // w1 <- w2, w2 <- w, w <- (v - oldeps w1 - delta w2) / gamma
            w1.copy_from_slice(w2);
            w2.copy_from_slice(w);
            for i in 0..n {
                w[i] = (v[i] - w1[i].scale(oldeps) - w2[i].scale(delta)).scale(denom);
            }
            par::axpy(T::from_f64(phi), w, x);
            let estimate = phibar * scale;
            if options.record_history {
                history.push(estimate);
            }
            if !estimate.is_finite() {
                flag = StopFlag::Breakdown;
                break 'restart;
            }
            if estimate <= tol || beta == 0.0 {
                // Verify against the true residual; restart the recurrence if it disagrees.
                verification_rounds += 1;
                if verification_rounds > options.max_verification_rounds + 1 {
                    flag = StopFlag::Stagnation;
                    break 'restart;
                }
                continue 'restart;
            }
        }
        flag = StopFlag::MaxIterations;
        break 'restart;
    }

    if flag != StopFlag::Converged {
        if let Some(best_x) = best.as_deref() {
            // Compare the last iterate against the best recorded one.
            ops.apply_operator(x, y).await?;
            counters.operator += 1;
            par::sub_into(b, y, r1);
            let last = par::norm2(r1);
            if last > best_residual {
                x.copy_from_slice(best_x);
                iteration = best_iteration;
            }
        }
        ops.apply_operator(x, y).await?;
        counters.operator += 1;
        par::sub_into(b, y, r1);
        true_residual = par::norm2(r1);
    }
    Ok(KrylovReport {
        flag,
        iterations: iteration,
        relative_residual: relative(true_residual, rhs_norm),
        residual_norm: true_residual,
        rhs_norm,
        residual_history: history,
        restart_position: None,
        operator_applications: counters.operator,
        preconditioner_applications: counters.preconditioner,
    })
}
