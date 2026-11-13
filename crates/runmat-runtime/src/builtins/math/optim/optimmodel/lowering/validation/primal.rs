//! Original floating-DAG feasibility. Row normalization uses its actual
//! evaluated operands, not a possibly cancelled assembled coefficient scale.
use super::*;
use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
use runmat_value::optimization_variable::OptimizationBoundRole as Bound;
fn close(left: f64, right: f64, equality: bool, tolerance: f64) -> bool {
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    if !equality && left <= right {
        return true;
    }
    let scale = left.abs().max(right.abs()).max(1.);
    let residual = if equality {
        (left / scale - right / scale).abs()
    } else {
        left / scale - right / scale
    };
    residual <= tolerance * (1. / scale + left.abs() / scale + right.abs() / scale)
}
impl Validator<'_> {
    pub(super) async fn primal(
        &self,
        candidate: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.primal_at_tolerance(
            candidate,
            super::super::super::super::linprog::compiled::validation_tolerance(),
            cooperation,
        )
        .await
    }
    pub(super) async fn primal_at_tolerance(
        &self,
        candidate: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(point) = &self.point else {
            return Ok(false);
        };
        self.primal_measured(point, candidate, tolerance, cooperation)
            .await
    }
    pub(super) async fn primal_measured(
        &self,
        point: &numerical::Point,
        candidate: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let policy = self.policy;
        for variable in &self.program.layout.variables {
            for row in 0..variable.count {
                let value = candidate[variable.start + row];
                let data = variable.view.definition();
                let lower = data
                    .bound_value(Bound::Lower, row)
                    .ok_or_else(|| policy.invalid("original lower bound is unavailable"))?;
                let upper = data
                    .bound_value(Bound::Upper, row)
                    .ok_or_else(|| policy.invalid("original upper bound is unavailable"))?;
                if lower == f64::INFINITY
                    || upper == f64::NEG_INFINITY
                    || (lower.is_finite() && !close(lower, value, false, tolerance))
                    || (upper.is_finite() && !close(value, upper, false, tolerance))
                {
                    return Ok(false);
                }
                cooperation.completed(1).await?;
            }
        }
        for constraint in &self.formulation.constraints {
            let left = &self
                .program
                .source
                .nodes
                .get(&constraint.left)
                .expect("original constraint root")
                .node()
                .shape;
            let right = &self
                .program
                .source
                .nodes
                .get(&constraint.right)
                .expect("original constraint root")
                .node()
                .shape;
            let plan = crate::builtins::common::broadcast::BroadcastPlan::new(left, right)
                .map_err(|e| policy.invalid(e))?;
            for (_, a, b) in plan.iter() {
                let left = point.at(constraint.left, a, policy)?;
                let right = point.at(constraint.right, b, policy)?;
                let valid = match constraint.relation {
                    Relation::Equal => close(left, right, true, tolerance),
                    Relation::LessEqual => close(left, right, false, tolerance),
                    Relation::GreaterEqual => close(right, left, false, tolerance),
                };
                if !valid {
                    return Ok(false);
                }
                cooperation.completed(1).await?;
            }
        }
        if let Some(objective) = &self.formulation.objective {
            if !point.at(objective.root, 0, policy)?.is_finite() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
