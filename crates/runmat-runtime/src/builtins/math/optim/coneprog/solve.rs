//! Cooperative transitions retain the original barrier/relaxation arithmetic.
use super::*;
use crate::builtins::math::optim::common::ScalarCooperation;

pub(super) async fn solve_coneprog(
    problem: &ConeProgram,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeOutcome> {
    cooperation.completed(0).await?;
    if let Some(reason) = impossible_bounds(problem) {
        return Ok(ConeOutcome::infeasible(problem, reason));
    }
    if problem.soc.is_empty()
        && problem.a_ineq.is_empty()
        && problem.a_eq.is_empty()
        && problem.lb.iter().all(|v| !v.is_finite())
        && problem.ub.iter().all(|v| !v.is_finite())
        && norm(&problem.f) > problem.options.tol
    {
        return Ok(ConeOutcome::unbounded(problem));
    }

    let mut linear_rows = problem.a_ineq.clone();
    let mut linear_rhs = problem.b_ineq.clone();
    append_bound_rows(problem, &mut linear_rows, &mut linear_rhs);

    let Some(mut x) = initial_strict_point(problem, &linear_rows, &linear_rhs, cooperation).await?
    else {
        return Ok(ConeOutcome::infeasible(problem, "No feasible point found."));
    };
    if constraint_violation(problem, &x) > 1.0e-5 {
        return Ok(ConeOutcome::infeasible(problem, "No feasible point found."));
    }
    if !is_strictly_feasible(problem, &linear_rows, &linear_rhs, &x) {
        relax_into_interior(problem, &linear_rows, &linear_rhs, &mut x, cooperation).await?;
    }
    if !is_strictly_feasible(problem, &linear_rows, &linear_rhs, &x)
        && norm(&problem.f) <= problem.options.tol
    {
        let fval = dot(&problem.f, &x);
        return finish_optimal(problem, x, fval, 0, cooperation).await;
    }
    if !is_strictly_feasible(problem, &linear_rows, &linear_rhs, &x) {
        return Ok(ConeOutcome::infeasible(
            problem,
            "No strict feasible point found.",
        ));
    }

    let mut iterations = 0usize;
    let outer = 9usize;
    let inner = (problem.options.max_iter / outer.max(1)).max(8);
    let mut t = 1.0;
    for _ in 0..outer {
        for _ in 0..inner {
            cooperation.completed(0).await?;
            iterations = iterations.checked_add(1).ok_or_else(|| {
                coneprog_error(
                    &ERROR_INVALID_INPUT,
                    "coneprog: iteration count exceeds platform limits",
                )
            })?;
            let grad = barrier_gradient(problem, &linear_rows, &linear_rhs, &x, t);
            let projected = project_gradient(&problem.a_eq, &grad);
            cooperation.completed(1).await?;
            let grad_norm = norm(&projected);
            if grad_norm <= problem.options.tol {
                break;
            }
            let current = barrier_value(problem, &linear_rows, &linear_rhs, &x, t);
            let mut step = 1.0 / (1.0 + grad_norm);
            let mut accepted = false;
            for _ in 0..40 {
                let mut candidate = x
                    .iter()
                    .zip(&projected)
                    .map(|(xi, gi)| xi - step * gi)
                    .collect::<Vec<_>>();
                cooperation.completed(0).await?;
                project_to_equalities(&problem.a_eq, &problem.b_eq, &mut candidate);
                cooperation.completed(1).await?;
                if is_strictly_feasible(problem, &linear_rows, &linear_rhs, &candidate)
                    && barrier_value(problem, &linear_rows, &linear_rhs, &candidate, t)
                        <= current - 1.0e-4 * step * grad_norm * grad_norm
                {
                    x = candidate;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        t *= 10.0;
    }

    let violation = constraint_violation(problem, &x);
    if violation > 1.0e-4 {
        return Ok(ConeOutcome::infeasible(problem, "No feasible point found."));
    }
    let fval = dot(&problem.f, &x);
    finish_optimal(problem, x, fval, iterations, cooperation).await
}

async fn initial_strict_point(
    problem: &ConeProgram,
    linear_rows: &[Vec<f64>],
    linear_rhs: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    cooperation.completed(0).await?;
    let n = problem.f.len();
    let mut x = if problem.a_eq.is_empty() {
        vec![0.0; n]
    } else {
        let result = pseudo_solve(&problem.a_eq, &problem.b_eq, n);
        cooperation.completed(1).await?;
        let Some(result) = result else {
            return Ok(None);
        };
        result
    };
    for i in 0..n {
        x[i] = match (problem.lb[i].is_finite(), problem.ub[i].is_finite()) {
            (true, true) => 0.5 * (problem.lb[i] + problem.ub[i]),
            (true, false) if x[i] <= problem.lb[i] => problem.lb[i] + 1.0,
            (false, true) if x[i] >= problem.ub[i] => problem.ub[i] - 1.0,
            _ => x[i],
        };
        cooperation.completed(1).await?;
    }
    cooperation.completed(0).await?;
    project_to_equalities(&problem.a_eq, &problem.b_eq, &mut x);
    cooperation.completed(1).await?;
    relax_into_interior(problem, linear_rows, linear_rhs, &mut x, cooperation).await?;
    Ok(Some(x))
}

async fn relax_into_interior(
    problem: &ConeProgram,
    linear_rows: &[Vec<f64>],
    linear_rhs: &[f64],
    x: &mut [f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    for _ in 0..800 {
        cooperation.completed(0).await?;
        let mut changed = false;
        for (row, rhs) in linear_rows.iter().zip(linear_rhs) {
            let slack = rhs - dot(row, x);
            if slack <= STRICT_MARGIN {
                let denom = dot(row, row).max(STRICT_MARGIN);
                let correction = (STRICT_MARGIN - slack) / denom;
                for (xi, ai) in x.iter_mut().zip(row) {
                    *xi -= correction * ai;
                }
                changed = true;
            }
            cooperation.completed(1).await?;
        }
        for constraint in &problem.soc {
            let z = cone_z(constraint, x);
            let z_norm = norm(&z);
            let q = dot(&constraint.d, x) + constraint.gamma;
            let violation = z_norm - q + STRICT_MARGIN;
            if violation > 0.0 {
                let grad = cone_violation_gradient(constraint, &z, z_norm);
                let denom = dot(&grad, &grad).max(STRICT_MARGIN);
                for (xi, gi) in x.iter_mut().zip(&grad) {
                    *xi -= violation * gi / denom;
                }
                changed = true;
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(0).await?;
        project_to_equalities(&problem.a_eq, &problem.b_eq, x);
        cooperation.completed(1).await?;
        if !changed {
            break;
        }
    }
    Ok(())
}

async fn finish_optimal(
    problem: &ConeProgram,
    x: Vec<f64>,
    fval: f64,
    iterations: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeOutcome> {
    cooperation.completed(0).await?;
    // The KKT multiplier SVD keeps its existing arithmetic and atomic boundary.
    let outcome = optimal_outcome(problem, x, fval, iterations);
    cooperation.completed(1).await?;
    Ok(outcome)
}
