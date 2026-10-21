//! Restarted GMRES(m) with left, right or flexible (FGMRES) preconditioning.
//!
//! Orthogonalization is classical Gram–Schmidt with one unconditional reorthogonalization pass
//! (CGS2): every coefficient is a deterministic blocked dot product, so the basis is reproducible
//! under threads while matching modified Gram–Schmidt stability.

use super::{relative, split_work, tolerance, ApplyStatus, KrylovOps, KrylovOptions, KrylovReport, StopFlag};
use crate::error::SolverResult;
use crate::par;
use crate::scalar::Scalar;

/// Where the preconditioner is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerSide {
    /// Minimizes `‖M⁻¹(b − A x)‖` (MATLAB `gmres`).
    Left,
    /// Minimizes the true residual `‖b − A M⁻¹ u‖`.
    Right,
    /// Right preconditioning that stores `M⁻¹ v_j`, so `M` may change between applications
    /// (e.g. an inner iterative solve or a nonlinear multigrid cycle).
    Flexible,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GmresOptions {
    pub restart: usize,
    pub side: PreconditionerSide,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            restart: 50,
            side: PreconditionerSide::Right,
        }
    }
}

pub fn gmres_workspace_len(n: usize, options: &KrylovOptions, gmres: &GmresOptions) -> usize {
    let m = gmres.restart.max(1);
    let flexible = if gmres.side == PreconditionerSide::Flexible { m } else { 0 };
    n * ((m + 1) + flexible + 3 + usize::from(options.return_best_iterate))
}

fn givens<T: Scalar>(a: T, b: T) -> (f64, T, T) {
    // Returns (c, s, rho) with [c s; -conj(s) c] [a; b] = [rho; 0].
    let (abs_a, abs_b) = (a.abs(), b.abs());
    if abs_b == 0.0 {
        return (1.0, T::zero(), a);
    }
    if abs_a == 0.0 {
        return (0.0, b.conj().scale(1.0 / abs_b), T::from_f64(abs_b));
    }
    let r = abs_a.hypot(abs_b);
    let phase = a.scale(1.0 / abs_a);
    let c = abs_a / r;
    let s = phase * b.conj().scale(1.0 / r);
    (c, s, phase.scale(r))
}

pub async fn gmres_async<T: Scalar, O: KrylovOps<T>>(
    ops: &mut O,
    b: &[T],
    x: &mut [T],
    options: &KrylovOptions,
    gmres: &GmresOptions,
    work: &mut [T],
) -> SolverResult<KrylovReport> {
    let n = b.len();
    let m = gmres.restart.max(1).min(n.max(1));
    let side = if ops.has_preconditioner() {
        gmres.side
    } else {
        PreconditionerSide::Right
    };
    let flexible = side == PreconditionerSide::Flexible;
    let total = (m + 1) + if flexible { m } else { 0 } + 3 + usize::from(options.return_best_iterate);
    let mut parts = split_work(work, n, total);
    let mut best = if options.return_best_iterate { parts.pop() } else { None };
    let r = parts.pop().unwrap();
    let w = parts.pop().unwrap();
    let t = parts.pop().unwrap();
    let z_basis: Vec<&mut [T]> = if flexible {
        parts.split_off(m + 1)
    } else {
        Vec::new()
    };
    let mut z_basis = z_basis;
    let mut basis = parts; // m + 1 vectors

    let mut counters = super::Counters {
        operator: 0,
        preconditioner: 0,
    };
    let rhs_norm = par::norm2(b);
    let tol = tolerance(options, rhs_norm);
    let mut history = Vec::new();
    let mut hessenberg = vec![T::zero(); (m + 1) * m];
    let mut cs = vec![0.0f64; m];
    let mut sn = vec![T::zero(); m];
    let mut g = vec![T::zero(); m + 1];
    let mut coefficients = vec![T::zero(); m + 1];
    let mut iterations = 0usize;
    let mut outer = 0usize;
    let flag;
    let (mut best_residual, mut best_iteration) = (f64::INFINITY, 0usize);
    let mut previous_residual = f64::INFINITY;
    let mut stagnant = 0usize;
    let mut true_residual;
    let mut position = (0usize, 0usize);

    loop {
        // True residual r = b - A x.
        if x.iter().all(|value| *value == T::zero()) {
            r.copy_from_slice(b);
        } else {
            ops.apply_operator(x, t).await?;
            counters.operator += 1;
            par::sub_into(b, t, r);
        }
        true_residual = par::norm2(r);
        if options.record_history && history.is_empty() {
            history.push(true_residual);
        }
        if true_residual < best_residual {
            best_residual = true_residual;
            best_iteration = iterations;
            if let Some(best) = best.as_deref_mut() {
                best.copy_from_slice(x);
            }
        }
        if rhs_norm == 0.0 {
            x.iter_mut().for_each(|value| *value = T::zero());
            true_residual = 0.0;
            flag = StopFlag::Converged;
            break;
        }
        if true_residual <= tol {
            flag = StopFlag::Converged;
            break;
        }
        if iterations >= options.max_iterations {
            flag = StopFlag::MaxIterations;
            break;
        }
        if true_residual >= previous_residual * (1.0 - 1.0e-14) {
            stagnant += 1;
            if stagnant >= options.max_stagnation_steps {
                flag = StopFlag::Stagnation;
                break;
            }
        } else {
            stagnant = 0;
        }
        previous_residual = true_residual;
        outer += 1;

        // Starting vector (left: preconditioned residual).
        if side == PreconditionerSide::Left {
            counters.preconditioner += 1;
            if ops.apply_preconditioner(r, basis[0]).await? == ApplyStatus::NumericalFailure {
                flag = StopFlag::PreconditionerFailure;
                break;
            }
        } else {
            basis[0].copy_from_slice(r);
        }
        let beta = par::norm2(basis[0]);
        if !beta.is_finite() {
            flag = StopFlag::PreconditionerFailure;
            break;
        }
        if beta == 0.0 {
            flag = StopFlag::Converged;
            break;
        }
        // Scale the inner estimate into true-residual units.
        let scale = true_residual / beta;
        par::scale(T::from_f64(1.0 / beta), basis[0]);
        g.iter_mut().for_each(|value| *value = T::zero());
        g[0] = T::from_f64(beta);
        hessenberg.iter_mut().for_each(|value| *value = T::zero());
        let mut columns = 0usize;
        let mut inner_failed = None;
        for j in 0..m {
            if iterations >= options.max_iterations {
                break;
            }
            iterations += 1;
            // w = A M⁻¹ v_j (right/flexible), M⁻¹ A v_j (left), A v_j (none).
            match side {
                PreconditionerSide::Right if ops.has_preconditioner() => {
                    counters.preconditioner += 1;
                    if ops.apply_preconditioner(basis[j], t).await? == ApplyStatus::NumericalFailure {
                        inner_failed = Some(StopFlag::PreconditionerFailure);
                        iterations -= 1;
                        break;
                    }
                    ops.apply_operator(t, w).await?;
                    counters.operator += 1;
                }
                PreconditionerSide::Flexible => {
                    counters.preconditioner += 1;
                    if ops.apply_preconditioner(basis[j], z_basis[j]).await?
                        == ApplyStatus::NumericalFailure
                    {
                        inner_failed = Some(StopFlag::PreconditionerFailure);
                        iterations -= 1;
                        break;
                    }
                    ops.apply_operator(z_basis[j], w).await?;
                    counters.operator += 1;
                }
                PreconditionerSide::Left => {
                    ops.apply_operator(basis[j], t).await?;
                    counters.operator += 1;
                    counters.preconditioner += 1;
                    if ops.apply_preconditioner(t, w).await? == ApplyStatus::NumericalFailure {
                        inner_failed = Some(StopFlag::PreconditionerFailure);
                        iterations -= 1;
                        break;
                    }
                }
                PreconditionerSide::Right => {
                    ops.apply_operator(basis[j], w).await?;
                    counters.operator += 1;
                }
            }
            // CGS2 against v_0..v_j.
            for pass in 0..2 {
                for i in 0..=j {
                    coefficients[i] = par::dot(basis[i], w);
                }
                for i in 0..=j {
                    par::axpy(-coefficients[i], basis[i], w);
                    let entry = &mut hessenberg[j * (m + 1) + i];
                    *entry = if pass == 0 { coefficients[i] } else { *entry + coefficients[i] };
                }
            }
            let h_next = par::norm2(w);
            hessenberg[j * (m + 1) + j + 1] = T::from_f64(h_next);
            if !h_next.is_finite() {
                inner_failed = Some(StopFlag::Breakdown);
                break;
            }
            // Apply previous rotations to column j.
            for i in 0..j {
                let (a, bb) = (hessenberg[j * (m + 1) + i], hessenberg[j * (m + 1) + i + 1]);
                hessenberg[j * (m + 1) + i] = a.scale(cs[i]) + sn[i] * bb;
                hessenberg[j * (m + 1) + i + 1] = -(sn[i].conj() * a) + bb.scale(cs[i]);
            }
            let (c, s, rho) = givens(hessenberg[j * (m + 1) + j], hessenberg[j * (m + 1) + j + 1]);
            cs[j] = c;
            sn[j] = s;
            hessenberg[j * (m + 1) + j] = rho;
            hessenberg[j * (m + 1) + j + 1] = T::zero();
            let gj = g[j];
            g[j] = gj.scale(c);
            g[j + 1] = -(s.conj() * gj);
            columns = j + 1;
            let estimate = g[j + 1].abs() * scale;
            if options.record_history {
                history.push(estimate);
            }
            ops.checkpoint(iterations, relative(estimate, rhs_norm)).await?;
            position = (outer, j + 1);
            if estimate <= tol || h_next <= f64::EPSILON * beta {
                break;
            }
            if j + 1 < m + 1 {
                let (head, tail) = basis.split_at_mut(j + 1);
                let _ = head;
                tail[0].copy_from_slice(w);
                par::scale(T::from_f64(1.0 / h_next), tail[0]);
            }
        }
        if columns > 0 {
            // Solve the upper-triangular least-squares system H y = g.
            let mut y = vec![T::zero(); columns];
            for i in (0..columns).rev() {
                let mut sum = g[i];
                for k in i + 1..columns {
                    sum -= hessenberg[k * (m + 1) + i] * y[k];
                }
                let diagonal = hessenberg[i * (m + 1) + i];
                if diagonal.abs() == 0.0 {
                    inner_failed = Some(StopFlag::Breakdown);
                    break;
                }
                y[i] = sum / diagonal;
            }
            if inner_failed != Some(StopFlag::Breakdown) {
                match side {
                    PreconditionerSide::Flexible => {
                        for (k, coefficient) in y.iter().enumerate() {
                            par::axpy(*coefficient, z_basis[k], x);
                        }
                    }
                    PreconditionerSide::Right if ops.has_preconditioner() => {
                        w.iter_mut().for_each(|value| *value = T::zero());
                        for (k, coefficient) in y.iter().enumerate() {
                            par::axpy(*coefficient, basis[k], w);
                        }
                        counters.preconditioner += 1;
                        if ops.apply_preconditioner(w, t).await? == ApplyStatus::NumericalFailure {
                            flag = StopFlag::PreconditionerFailure;
                            break;
                        }
                        par::axpy(T::one(), t, x);
                    }
                    _ => {
                        for (k, coefficient) in y.iter().enumerate() {
                            par::axpy(*coefficient, basis[k], x);
                        }
                    }
                }
            }
        }
        if let Some(failure) = inner_failed {
            flag = failure;
            break;
        }
    }

    if flag != StopFlag::Converged {
        ops.apply_operator(x, t).await?;
        counters.operator += 1;
        par::sub_into(b, t, r);
        true_residual = par::norm2(r);
        if let Some(best_x) = best.as_deref() {
            if best_residual < true_residual {
                x.copy_from_slice(best_x);
                true_residual = best_residual;
                iterations = best_iteration;
            }
        }
    }
    Ok(KrylovReport {
        flag,
        iterations,
        relative_residual: relative(true_residual, rhs_norm),
        residual_norm: true_residual,
        rhs_norm,
        residual_history: history,
        restart_position: Some(position),
        operator_applications: counters.operator,
        preconditioner_applications: counters.preconditioner,
    })
}
