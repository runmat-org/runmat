//! One original generation per solver probe, with no synthetic user callable.
use super::*;
use compiled::{EvaluationStop, Oracle, Probe, ProbeRole};
use futures::{future::LocalBoxFuture, FutureExt};
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;

pub(super) struct Original<'a> {
    pub model: &'a Model,
    pub formulation: &'a formulation::Formulation,
    pub policy: Policy,
}
impl Oracle for Original<'_> {
    fn evaluate<'a>(
        &'a mut self,
        x: &'a [f64],
        role: ProbeRole,
        c: &'a mut ScalarCooperation,
    ) -> LocalBoxFuture<'a, BuiltinResult<Result<Probe, EvaluationStop>>> {
        async move {
            let (roots, selected): (&[GcHandle], _) = match role {
                ProbeRole::Full => (&self.formulation.roots, None),
                ProbeRole::ConstraintsOnly => {
                    (&self.model.constraints.roots, Some(&self.model.constraints))
                }
            };
            let Some(point) = numerical::Point::evaluate_layout(
                &self.model.source,
                &self.model.layout,
                selected,
                roots,
                x,
                self.policy,
                c,
            )
            .await?
            else {
                return Ok(Err(EvaluationStop::Undefined));
            };
            let mut inequality = runmat_value::exact_capacity_vec(self.model.inequalities)
                .map_err(|e| self.policy.storage(e))?;
            let mut equality = runmat_value::exact_capacity_vec(self.model.equalities)
                .map_err(|e| self.policy.storage(e))?;
            for (constraint, geometry) in self
                .formulation
                .constraints
                .iter()
                .zip(&self.model.geometries)
            {
                let values = if constraint.relation == Relation::Equal {
                    &mut equality
                } else {
                    &mut inequality
                };
                for row in 0..geometry.plan.len() {
                    let (a, b) = geometry.indices(row).ok_or_else(|| {
                        self.policy
                            .invalid("original constraint row is unavailable")
                    })?;
                    let left = point.at(constraint.left, a, self.policy)?;
                    let right = point.at(constraint.right, b, self.policy)?;
                    let residual = if constraint.relation == Relation::GreaterEqual {
                        right - left
                    } else {
                        left - right
                    };
                    if !residual.is_finite() {
                        return Ok(Err(EvaluationStop::Undefined));
                    }
                    values.push(residual);
                    c.completed(1).await?;
                }
            }
            Ok(Ok(match role {
                ProbeRole::ConstraintsOnly => Probe::ConstraintsOnly {
                    inequality,
                    equality,
                },
                ProbeRole::Full => {
                    let objective = match &self.formulation.objective {
                        Some(objective) => {
                            let value = point.at(objective.root, 0, self.policy)?;
                            if objective.sense == Sense::Maximize {
                                -value
                            } else {
                                value
                            }
                        }
                        None => 0.,
                    };
                    if !objective.is_finite() {
                        return Ok(Err(EvaluationStop::Undefined));
                    }
                    Probe::Full {
                        objective,
                        inequality,
                        equality,
                    }
                }
            }))
        }
        .boxed_local()
    }
}
