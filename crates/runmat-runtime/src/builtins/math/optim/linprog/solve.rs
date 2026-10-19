//! Numerical ordering is unchanged; invocation budgets own cooperative turns.
use super::combinations::Combinations;
use super::*;
use crate::builtins::math::optim::common::ScalarCooperation;

pub(super) async fn solve_linprog(
    problem: &LinearProgram,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<LinprogOutcome> {
    cooperation.completed(0).await?;
    // nalgebra SVD remains an atomic numerical primitive, not hard-preemptible work.
    let face = equality_face(problem, problem.f.len());
    cooperation.completed(1).await?;
    let Some(face) = face else {
        return Ok(LinprogOutcome::infeasible("No feasible point found."));
    };
    let reduced = reduce_to_equality_face(problem, &face);
    cooperation.completed(1).await?;
    let k = reduced.f.len();
    let mut candidates = Vec::new();
    let mut combinations = 0usize;
    for active_count in 0..=reduced.a_ineq.len().min(k) {
        let mut traversal = Combinations::new(reduced.a_ineq.len(), active_count);
        while let Some(active) = traversal.next() {
            cooperation.completed(0).await?;
            let mut rows = reduced.a_eq.clone();
            let mut rhs = reduced.b_eq.clone();
            for &index in active {
                rows.push(reduced.a_ineq[index].clone());
                rhs.push(reduced.b_ineq[index]);
            }
            let y = pseudo_solve(&rows, &rhs, k);
            cooperation.completed(1).await?;
            if let Some(y) = y {
                combinations = combinations.checked_add(1).ok_or_else(|| {
                    linprog_error_with_detail(
                        &LINPROG_ERROR_INVALID_INPUT,
                        "candidate count exceeds platform limits",
                    )
                })?;
                if is_feasible(&reduced, &y) {
                    candidates.push(y);
                }
            }
        }
    }
    let feasible_fallback = if candidates.is_empty() {
        let y0 = vec![0.0; k];
        is_feasible(&reduced, &y0).then_some(y0)
    } else {
        None
    };
    if candidates.is_empty() && feasible_fallback.is_none() {
        return Ok(LinprogOutcome::infeasible("No feasible point found."));
    }
    if has_unbounded_descent_direction(&reduced, cooperation).await? {
        return Ok(LinprogOutcome::unbounded(combinations));
    }
    if let Some(y) = feasible_fallback {
        candidates.push(y);
    }
    let mut best_y = candidates[0].clone();
    let mut best_fval = dot(&reduced.f, &best_y);
    for candidate in candidates.into_iter().skip(1) {
        cooperation.completed(1).await?;
        let fval = dot(&reduced.f, &candidate);
        if fval < best_fval - TOL {
            best_y = candidate;
            best_fval = fval;
        }
    }
    let best = lift_from_equality_face(&face, &best_y);
    let best_fval = dot(&problem.f, &best);
    let constrviolation = constraint_violation(problem, &best);
    cooperation.completed(1).await?;
    Ok(LinprogOutcome {
        x: Some(best),
        fval: Some(best_fval),
        exitflag: 1,
        iterations: combinations,
        constrviolation,
        message: "Optimal solution found.".to_string(),
    })
}

async fn has_unbounded_descent_direction(
    problem: &LinearProgram,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let n = problem.f.len();
    for active_count in 0..=problem.a_ineq.len().min(n.saturating_sub(1)) {
        let mut traversal = Combinations::new(problem.a_ineq.len(), active_count);
        while let Some(active) = traversal.next() {
            cooperation.completed(0).await?;
            let mut rows = problem.a_eq.clone();
            for &index in active {
                rows.push(problem.a_ineq[index].clone());
            }
            let directions = candidate_nullspace_descent_directions(&rows, &problem.f, n);
            cooperation.completed(1).await?;
            for direction in directions {
                cooperation.completed(1).await?;
                if is_recession_direction(problem, &direction) && dot(&problem.f, &direction) < -TOL
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}
