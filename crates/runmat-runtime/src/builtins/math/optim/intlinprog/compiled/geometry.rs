//! Intrinsic typed geometry before raw workspace indexing or selected effects.
use super::*;
pub(super) async fn validate(problem: &Problem, c: &mut ScalarCooperation) -> BuiltinResult<()> {
    let n = problem.f.len();
    if n == 0
        || problem.lower.len() != n
        || problem.upper.len() != n
        || problem.a_ineq.n != n
        || problem.a_eq.n != n
        || problem.a_ineq.m != problem.b_ineq.len()
        || problem.a_eq.m != problem.b_eq.len()
    {
        return Err(invalid("compiled integer model dimensions disagree"));
    }
    for values in [&problem.f, &problem.b_ineq, &problem.b_eq] {
        for value in values {
            if !value.is_finite() {
                return Err(invalid(
                    "compiled integer objective and rows must be finite",
                ));
            }
            c.completed(1).await?;
        }
    }
    for (&lower, &upper) in problem.lower.iter().zip(&problem.upper) {
        if lower.is_nan() || upper.is_nan() {
            return Err(invalid("compiled integer bound is NaN"));
        }
        c.completed(1).await?;
    }
    let mut previous = None;
    for &column in &problem.integers {
        if column >= n || previous.is_some_and(|old| old >= column) {
            return Err(invalid(
                "compiled integer columns must be sorted, unique and in range",
            ));
        }
        previous = Some(column);
        c.completed(1).await?;
    }
    for matrix in [&problem.a_ineq, &problem.a_eq] {
        common::conic::matrix::validate_csc(matrix, c, |detail: &str| invalid(detail)).await?;
    }
    let mut previous = None;
    for semi in &problem.semis {
        if semi.column >= n
            || previous.is_some_and(|old| old >= semi.column)
            || problem.lower[semi.column] != 0.
            || problem.upper[semi.column] != semi.upper
        {
            return Err(invalid(
                "compiled semi domains disagree with their sorted convex hulls",
            ));
        }
        let validated = Semi::from_bounds(
            semi.column,
            semi.lower,
            semi.upper,
            problem.integers.binary_search(&semi.column).is_ok(),
        )?;
        if validated.lower != semi.lower {
            return Err(invalid("compiled semi-integer lower bound is not integral"));
        }
        previous = Some(semi.column);
        c.completed(1).await?;
    }
    Ok(())
}
