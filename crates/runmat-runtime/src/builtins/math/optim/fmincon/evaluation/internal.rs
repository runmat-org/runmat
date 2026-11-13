//! Borrowed numerical merit probes, not selected user callbacks. The outer
//! solver alone owns expensive-evaluation counts and public observations.
use super::{Evaluation, EvaluationStop, Evaluator, Point, Probe, ProbeRole};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use futures::future::LocalBoxFuture;

pub(in crate::builtins::math::optim) trait Oracle {
    fn evaluate<'a>(
        &'a mut self,
        x: &'a [f64],
        role: ProbeRole,
        cooperation: &'a mut ScalarCooperation,
    ) -> LocalBoxFuture<'a, BuiltinResult<std::result::Result<Probe, EvaluationStop>>>;
}
impl Evaluator<'_> {
    pub(super) async fn evaluate_internal(
        &mut self,
        x: &[f64],
        role: ProbeRole,
    ) -> BuiltinResult<std::result::Result<Probe, EvaluationStop>> {
        if self.semi_infinite.is_some()
            || self.attainment.is_some()
            || self.constraints.is_some()
            || self.objective_gradient
            || self.constraint_gradient
        {
            return Err(super::numeric::invalid(
                &self.policy,
                "internal merit",
                "requires its scalar numerical-derivative domain without selected callbacks",
            ));
        }
        if !self.budget.begin_point() {
            return Ok(Err(EvaluationStop::BudgetExhausted));
        }
        let returned = self
            .internal
            .as_deref_mut()
            .expect("selected internal oracle")
            .evaluate(x, role, self.cooperation)
            .await?;
        let returned = match returned {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        if !matches!(
            (&role, &returned),
            (ProbeRole::Full, Probe::Full { .. })
                | (ProbeRole::ConstraintsOnly, Probe::ConstraintsOnly { .. })
        ) {
            return Err(super::numeric::invalid(
                &self.policy,
                "internal merit",
                "changed its typed requested probe role",
            ));
        }
        if let Probe::Full { objective, .. } = &returned {
            if !objective.is_finite() {
                return Ok(Err(EvaluationStop::Undefined));
            }
        }
        for values in [returned.inequality(), returned.equality()] {
            for value in values {
                if !value.is_finite() {
                    return Ok(Err(EvaluationStop::Undefined));
                }
                self.cooperation.completed(1).await?;
            }
        }
        let extents = (returned.inequality().len(), returned.equality().len());
        if self
            .constraint_extents
            .is_some_and(|prior| prior != extents)
        {
            return Err(super::numeric::invalid(
                &self.policy,
                "internal merit",
                "changed its evaluated constraint extent",
            ));
        }
        // Only an actually returned typed probe completes this count. A
        // deadline/budget stop has no invented objective/constraint result.
        // These private counters never enter expensive outer funcCount.
        match role {
            ProbeRole::Full => {
                self.budget.completed_objective();
                if extents != (0, 0) {
                    self.budget.completed_constraints();
                }
            }
            ProbeRole::ConstraintsOnly => self.budget.completed_constraints(),
        }
        self.constraint_extents = Some(extents);
        Ok(Ok(returned))
    }

    pub(super) async fn internal_point(&mut self, x: &[f64]) -> BuiltinResult<Evaluation> {
        let probe = self.evaluate_internal(x, ProbeRole::Full).await?;
        let probe = match probe {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
        let Probe::Full {
            objective,
            inequality,
            equality,
        } = probe
        else {
            return Err(super::numeric::invalid(
                &self.policy,
                "internal merit",
                "full point requires an actual modeled objective",
            ));
        };
        let inequality_jacobian = inequality
            .is_empty()
            .then(|| CscMatrix::zeros((0, x.len())));
        let equality_jacobian = equality.is_empty().then(|| CscMatrix::zeros((0, x.len())));
        Ok(Ok(Point {
            semi_infinite: None,
            objectives: None,
            f: objective,
            inequality,
            equality,
            gradient: None,
            inequality_jacobian,
            equality_jacobian,
            projected_objective: false,
            projected_constraints: false,
        }))
    }
}
