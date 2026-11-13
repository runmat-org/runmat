//! Bound-feasible finite differences. A complete sweep has a known point cost;
//! it cannot overrun the original evaluator's hard point budget. Already-owned
//! analytic derivatives and accepted function values are never replayed to fill
//! outputs.
use super::evaluation::{EvaluationStop, Evaluator, Point, ProbeRole};
use super::workspace;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;

pub(super) use runmat_builtins::OptimDifferenceType as Difference;
pub(super) enum Completion {
    Complete,
    Undefined,
    BudgetExhausted,
    Time,
}

use crate::builtins::math::optim::common::derivative_scaling::{
    DerivativeScaling, ProbePoints as Perturbation,
};
fn perturbation(
    scaling: &DerivativeScaling,
    name: &str,
    x: f64,
    variable: usize,
    lower: f64,
    upper: f64,
    mode: Difference,
) -> BuiltinResult<Option<Perturbation>> {
    scaling.points(name, x, variable, lower, upper, mode == Difference::Central)
}

pub(super) async fn sweep_cost(
    evaluator: &mut Evaluator<'_>,
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    mode: Difference,
) -> BuiltinResult<usize> {
    if evaluator.objective_gradient
        && (evaluator.constraints.is_none() || evaluator.constraint_gradient)
    {
        return Ok(0);
    }
    let mut count = 0usize;
    for variable in 0..evaluator.original_variables(x.len()) {
        if let Some(perturbation) = perturbation(
            &evaluator.scaling,
            evaluator.policy.owner.name,
            x[variable],
            variable,
            lower[variable],
            upper[variable],
            mode,
        )? {
            count = count.saturating_add(if perturbation.second.is_some() { 2 } else { 1 });
        }
        evaluator.cooperation.completed(1).await?;
    }
    Ok(count)
}

fn derivative(
    base: f64,
    first: f64,
    second: Option<f64>,
    h: f64,
    second_h: Option<f64>,
) -> Option<f64> {
    let value = match (second, second_h) {
        (Some(second), Some(k)) => {
            // h>0, k<0 for central differences. Weights use actual feasible
            // representable points, including a truncated step near a bound.
            let total = h - k;
            (-k / total) * ((first - base) / h) + (h / total) * ((second - base) / k)
        }
        _ => (first - base) / h,
    };
    value.is_finite().then_some(value)
}

pub(super) async fn complete(
    evaluator: &mut Evaluator<'_>,
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    point: &mut Point,
    mode: Difference,
) -> BuiltinResult<Completion> {
    if evaluator.time_limit_reached() {
        return Ok(Completion::Time);
    }
    let objective = point.gradient.is_none();
    let constraints = point.inequality_jacobian.is_none() || point.equality_jacobian.is_none();
    if !objective && !constraints {
        return Ok(Completion::Complete);
    }
    let owner = evaluator.policy.owner;
    let n = x.len();
    let original_variables = evaluator.original_variables(n);
    let mut perturbations = workspace::capacity(n, owner)?;
    let mut cost = 0usize;
    let mut projected = false;
    for variable in 0..n {
        let perturbation = if variable < original_variables {
            perturbation(
                &evaluator.scaling,
                evaluator.policy.owner.name,
                x[variable],
                variable,
                lower[variable],
                upper[variable],
                mode,
            )?
        } else {
            None
        };
        if let Some(perturbation) = perturbation {
            cost = cost.saturating_add(if perturbation.second.is_some() { 2 } else { 1 });
        }
        projected |= variable < original_variables && perturbation.is_none();
        perturbations.push(perturbation);
        evaluator.cooperation.completed(1).await?;
    }
    if cost > evaluator.budget.remaining() {
        return Ok(Completion::BudgetExhausted);
    }
    let mut gradient = if objective {
        Some(workspace::capacity(n, owner)?)
    } else {
        None
    };
    let mut inequality_offsets = workspace::capacity(n.saturating_add(1), owner)?;
    let mut equality_offsets = workspace::capacity(n.saturating_add(1), owner)?;
    let mut inequality_rows = Vec::new();
    let mut inequality_values = Vec::new();
    let mut equality_rows = Vec::new();
    let mut equality_values = Vec::new();
    inequality_offsets.push(0);
    equality_offsets.push(0);
    let mut trial = workspace::capacity(n, owner)?;
    for &value in x {
        trial.push(value);
        evaluator.cooperation.completed(1).await?;
    }
    for (variable, perturbation) in perturbations.into_iter().enumerate() {
        let pair = if let Some(perturbation) = perturbation {
            trial[variable] = perturbation.first;
            let first = match evaluator
                .evaluate_probe(
                    &trial,
                    if objective {
                        ProbeRole::Full
                    } else {
                        ProbeRole::ConstraintsOnly
                    },
                )
                .await?
            {
                Ok(value) => value,
                Err(EvaluationStop::Undefined) => return Ok(Completion::Undefined),
                Err(EvaluationStop::Time) => return Ok(Completion::Time),
                Err(EvaluationStop::BudgetExhausted) => return Ok(Completion::BudgetExhausted),
            };
            let second = if let Some(value) = perturbation.second {
                trial[variable] = value;
                match evaluator
                    .evaluate_probe(
                        &trial,
                        if objective {
                            ProbeRole::Full
                        } else {
                            ProbeRole::ConstraintsOnly
                        },
                    )
                    .await?
                {
                    Ok(value) => Some(value),
                    Err(EvaluationStop::Undefined) => return Ok(Completion::Undefined),
                    Err(EvaluationStop::Time) => return Ok(Completion::Time),
                    Err(EvaluationStop::BudgetExhausted) => return Ok(Completion::BudgetExhausted),
                }
            } else {
                None
            };
            trial[variable] = x[variable];
            Some((first, second, perturbation))
        } else {
            None
        };
        if let Some(gradient) = gradient.as_mut() {
            let value = if let Some((first, second, perturbation)) = &pair {
                let Some(value) = derivative(
                    point.f,
                    first.actual_objective()?,
                    second.as_ref().map(|v| v.actual_objective()).transpose()?,
                    perturbation.first - x[variable],
                    perturbation.second.map(|v| v - x[variable]),
                ) else {
                    return Ok(Completion::Undefined);
                };
                value
            } else {
                0.0
            };
            gradient.push(value);
        }
        if constraints {
            for (base, offsets, rows, values, is_equality) in [
                (
                    &point.inequality,
                    &mut inequality_offsets,
                    &mut inequality_rows,
                    &mut inequality_values,
                    false,
                ),
                (
                    &point.equality,
                    &mut equality_offsets,
                    &mut equality_rows,
                    &mut equality_values,
                    true,
                ),
            ] {
                for (row, &base) in base.iter().enumerate() {
                    let value = if variable >= original_variables {
                        evaluator.attainment_derivative(row, is_equality)
                    } else if let Some((first, second, perturbation)) = &pair {
                        let first = if is_equality {
                            first.equality()[row]
                        } else {
                            first.inequality()[row]
                        };
                        let second = second.as_ref().map(|v| {
                            if is_equality {
                                v.equality()[row]
                            } else {
                                v.inequality()[row]
                            }
                        });
                        let Some(value) = derivative(
                            base,
                            first,
                            second,
                            perturbation.first - x[variable],
                            perturbation.second.map(|v| v - x[variable]),
                        ) else {
                            return Ok(Completion::Undefined);
                        };
                        value
                    } else {
                        0.0
                    };
                    if value != 0.0 {
                        workspace::push(rows, row, owner)?;
                        workspace::push(values, value, owner)?;
                    }
                    evaluator.cooperation.completed(1).await?;
                }
                offsets.push(rows.len());
            }
        }
        evaluator.cooperation.completed(1).await?;
    }
    if let Some(gradient) = gradient {
        point.gradient = Some(gradient);
        point.projected_objective = projected;
    }
    if constraints {
        point.projected_constraints =
            projected && (!point.inequality.is_empty() || !point.equality.is_empty());
        point.inequality_jacobian = Some(CscMatrix::new(
            point.inequality.len(),
            n,
            inequality_offsets,
            inequality_rows,
            inequality_values,
        ));
        point.equality_jacobian = Some(CscMatrix::new(
            point.equality.len(),
            n,
            equality_offsets,
            equality_rows,
            equality_values,
        ));
    }
    Ok(Completion::Complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_bound_steps_and_nonuniform_central_formula() {
        let step = perturbation(
            &DerivativeScaling::default(),
            "fmincon",
            1.0,
            0,
            0.0,
            1.0,
            Difference::Forward,
        )
        .unwrap()
        .unwrap();
        assert!(step.first < 1.0 && step.first >= 0.0 && step.second.is_none());
        assert!(perturbation(
            &DerivativeScaling::default(),
            "fmincon",
            2.0,
            0,
            2.0,
            2.0,
            Difference::Central
        )
        .unwrap()
        .is_none());
        let x = 2.0;
        let h = 0.01;
        let k = -0.003;
        let slope = derivative(
            x * x,
            (x + h) * (x + h),
            Some((x + k) * (x + k)),
            h,
            Some(k),
        )
        .unwrap();
        assert!((slope - 4.0).abs() < 1e-11);
        assert!(derivative(0.0, f64::INFINITY, None, 1.0, None).is_none());
    }
}
