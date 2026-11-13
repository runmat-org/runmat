//! Actual returned binary64 point, original floating roots and absolute
//! feasibility tolerance. Exact witness/cutoff authority remains separate.
use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;

use runmat_value::optimization_variable::OptimizationBoundRole as Bound;
async fn residual(
    left: f64,
    right: f64,
    equality: bool,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Rational>> {
    let Some(left) = Rational::from_float(left) else {
        return Ok(None);
    };
    let Some(right) = Rational::from_float(right) else {
        return Ok(None);
    };
    let value = left.sub(&right, policy, c).await?;
    Ok(Some(if equality {
        value.abs()
    } else if value.is_positive() {
        value
    } else {
        Rational::zero()
    }))
}
impl Validator<'_> {
    pub(super) async fn integer_publication(
        &mut self,
        values: &[f64],
        exact_values: &[BigRational],
        domains_admitted: bool,
        tolerance: f64,
        cutoff: Option<&BigRational>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<mip::Evaluation> {
        let policy = self.policy;
        let rejected = || mip::Evaluation {
            objective: f64::NAN,
            violation: f64::INFINITY,
            admitted: false,
        };
        if !self.prepare(values, c).await? {
            return Ok(rejected());
        }
        let point = self.point.as_ref().expect("original binary point");
        let mut violation = Rational::zero();
        for variable in &self.program.layout.variables {
            for row in 0..variable.count {
                let actual = values[variable.start + row];
                if variable.domain.is_semi() && actual == 0. {
                    c.completed(1).await?;
                    continue;
                }
                for (role, lower) in [(Bound::Lower, true), (Bound::Upper, false)] {
                    let limit = variable
                        .view
                        .definition()
                        .bound_value(role, row)
                        .ok_or_else(|| {
                            policy.invalid("original bound exceeds variable definition")
                        })?;
                    if limit
                        == if lower {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        }
                    {
                        continue;
                    }
                    let (left, right) = if lower {
                        (limit, actual)
                    } else {
                        (actual, limit)
                    };
                    let Some(value) = residual(left, right, false, policy, c).await? else {
                        return Ok(rejected());
                    };
                    if value.compare(&violation, policy, c).await?.is_gt() {
                        violation = value;
                    }
                }
                c.completed(1).await?;
            }
        }
        for constraint in &self.formulation.constraints {
            let left = &self.program.source.nodes[&constraint.left].node().shape;
            let right = &self.program.source.nodes[&constraint.right].node().shape;
            let plan = BroadcastPlan::new(left, right).map_err(|e| policy.invalid(e))?;
            for (_, a, b) in plan.iter() {
                let a = point.at(constraint.left, a, policy)?;
                let b = point.at(constraint.right, b, policy)?;
                let (a, b) = if constraint.relation == Relation::GreaterEqual {
                    (b, a)
                } else {
                    (a, b)
                };
                let Some(value) =
                    residual(a, b, constraint.relation == Relation::Equal, policy, c).await?
                else {
                    return Ok(rejected());
                };
                if value.compare(&violation, policy, c).await?.is_gt() {
                    violation = value;
                }
                c.completed(1).await?;
            }
        }
        let objective = match &self.formulation.objective {
            Some(objective) => {
                let value = point.at(objective.root, 0, policy)?;
                if objective.sense == Sense::Maximize {
                    -value
                } else {
                    value
                }
            }
            None => 0.,
        };
        let tolerance = Rational::from_float(tolerance)
            .ok_or_else(|| policy.invalid("integer tolerance is not finite"))?;
        let mut admitted = domains_admitted
            && objective.is_finite()
            && !violation.compare(&tolerance, policy, c).await?.is_gt();
        if let Some(cutoff) = cutoff {
            let Some(imported) = proof::imported(exact_values, policy, c).await? else {
                return Ok(rejected());
            };
            let actual = self
                .authority
                .integer_objective(&imported, c)
                .await?
                .export(policy, c)
                .await?;
            admitted &= &actual <= cutoff
                && BigRational::from_float(objective).is_some_and(|value| &value <= cutoff);
        }
        Ok(mip::Evaluation {
            objective,
            violation: mip::nonnegative(&violation.export(policy, c).await?),
            admitted,
        })
    }
}
