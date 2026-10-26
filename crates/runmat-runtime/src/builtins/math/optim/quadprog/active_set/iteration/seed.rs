//! Preserve represented feasible coordinates; reset only out-of-box starts.
use super::*;
pub(super) async fn prepare(problem: &mut Problem, c: &mut ScalarCooperation) -> BuiltinResult<()> {
    let missing = problem.x0.is_empty();
    if problem.x0.is_empty() {
        problem.x0 = zeros(problem.f.len(), c).await?;
    }
    if problem.x0.len() != problem.f.len() {
        return Err(invalid("active-set x0 differs from H column count"));
    }
    for ((x, &lower), &upper) in problem.x0.iter_mut().zip(&problem.lb).zip(&problem.ub) {
        if !x.is_finite() {
            return Err(invalid("active-set x0 must be finite"));
        }
        if *x < lower || *x > upper || missing && (*x == lower || *x == upper) {
            *x = if lower.is_finite() && upper.is_finite() {
                0.5 * lower + 0.5 * upper
            } else if lower.is_finite() {
                let next = lower.next_up();
                if next.is_finite() {
                    next
                } else {
                    lower
                }
            } else if upper.is_finite() {
                let next = upper.next_down();
                if next.is_finite() {
                    next
                } else {
                    upper
                }
            } else {
                *x
            };
            // An endpoint may have no finite representable interior neighbor.
            *x = x.max(lower).min(upper);
            if !x.is_finite() {
                return Err(invalid("no finite active-set bound start"));
            }
        }
        c.completed(1).await?;
    }
    Ok(())
}
