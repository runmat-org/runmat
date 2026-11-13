//! Continuous original domains, independently of the selected model controller.
use super::*;
use crate::builtins::math::optim::least_squares::LeastSquaresBounds;
use runmat_value::optimization_variable::OptimizationBoundRole;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn continuous_bounds(
    layout: &Layout,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<LeastSquaresBounds> {
    let bounds = continuous_bound_values(layout, policy, c).await?;
    for (&lower, &upper) in bounds.lower.iter().zip(&bounds.upper) {
        if lower.is_nan()
            || upper.is_nan()
            || lower > upper
            || lower == f64::INFINITY
            || upper == f64::NEG_INFINITY
        {
            return Err(policy
                .invalid("continuous bounds are inconsistent or have an invalid infinite side"));
        }
        c.completed(1).await?;
    }
    Ok(bounds)
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn continuous_bound_values(
    layout: &Layout,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<LeastSquaresBounds> {
    let mut lower =
        runmat_value::exact_capacity_vec(layout.columns).map_err(|e| policy.storage(e))?;
    let mut upper =
        runmat_value::exact_capacity_vec(layout.columns).map_err(|e| policy.storage(e))?;
    for &index in &layout.order {
        let variable = &layout.variables[index];
        if variable.domain != OptimizationVariableDomain::Continuous {
            return Err(policy.invalid("the selected continuous model solver cannot ignore an original integer variable domain"));
        }
        for row in 0..variable.count {
            lower.push(
                variable
                    .view
                    .definition()
                    .bound_value(OptimizationBoundRole::Lower, row)
                    .ok_or_else(|| policy.invalid("original lower bound exceeds its definition"))?,
            );
            upper.push(
                variable
                    .view
                    .definition()
                    .bound_value(OptimizationBoundRole::Upper, row)
                    .ok_or_else(|| policy.invalid("original upper bound exceeds its definition"))?,
            );
            c.completed(1).await?;
        }
    }
    Ok(LeastSquaresBounds { lower, upper })
}
