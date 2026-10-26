//! Compiled CSC admission before indexing the existing QP model owner.
use super::*;
pub(super) async fn validate(
    problem: &Problem,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let n = problem.f.len();
    if n == 0
        || problem.p.m != n
        || problem.p.n != n
        || problem.lb.len() != n
        || problem.ub.len() != n
        || problem.a_ineq.n != n
        || problem.a_eq.n != n
        || problem.a_ineq.m != problem.b_ineq.len()
        || problem.a_eq.m != problem.b_eq.len()
    {
        return Err(invalid("compiled quadratic model dimensions disagree"));
    }
    for values in [&problem.f, &problem.b_ineq, &problem.b_eq] {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| !value.is_finite()) {
                return Err(invalid("compiled quadratic costs and rows must be finite"));
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    for values in [&problem.lb, &problem.ub] {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| value.is_nan()) {
                return Err(invalid("compiled quadratic bounds cannot contain NaN"));
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    for matrix in [&problem.p, &problem.a_ineq, &problem.a_eq] {
        conic::matrix::validate_csc(matrix, cooperation, |detail: &str| invalid(detail)).await?;
    }
    for column in 0..problem.p.n {
        for entry in problem.p.colptr[column]..problem.p.colptr[column + 1] {
            if problem.p.rowval[entry] > column {
                return Err(invalid(
                    "compiled Hessian requires canonical upper-triangular storage",
                ));
            }
            cooperation.completed(1).await?;
        }
    }
    Ok(())
}
