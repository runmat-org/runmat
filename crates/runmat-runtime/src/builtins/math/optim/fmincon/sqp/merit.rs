//! Exact-penalty SQP line search over actual bound-feasible points. Only typed
//! numeric undefinedness shrinks a trial; callback/service errors propagate.
use super::super::{
    derivatives::{self, Completion, Difference},
    evaluation::{EvaluationStop, Evaluator, Point},
    model::{ConstraintValues, GeometryView, Linearization},
    workspace,
};
use crate::BuiltinResult;

pub(super) struct Accepted {
    pub x: Vec<f64>,
    pub point: Option<Point>,
    pub values: ConstraintValues,
    pub alpha: f64,
    pub step_norm: f64,
}
pub(super) type Result = std::result::Result<Accepted, SearchStop>;
pub(super) enum SearchStop {
    Time,
    Limit,
    NoDescent,
    StepBound,
}

pub(super) struct Search<'a, 'b> {
    pub evaluator: &'a mut Evaluator<'b>,
    pub geometry: &'a GeometryView<'a>,
    pub differences: Difference,
    pub penalty: f64,
    pub iteration: usize,
}
impl Search<'_, '_> {
    pub async fn search(
        &mut self,
        x: &[f64],
        point: &Point,
        linearization: &Linearization,
        direction: &[f64],
        multipliers: &[f64],
    ) -> BuiltinResult<Result> {
        let owner = self.evaluator.policy.owner;
        let (_, mut current_violation) =
            linearization.violation(self.evaluator.cooperation).await?;
        if !current_violation.is_finite() {
            return Ok(Err(SearchStop::NoDescent));
        }
        let mut activity = workspace::capacity(linearization.b.len(), owner)?;
        for &rhs in &linearization.b {
            activity.push(-rhs);
            self.evaluator.cooperation.completed(1).await?;
        }
        let mut slope = 0.0;
        let mut compensation = 0.0;
        let gradient = point.gradient.as_ref().expect("complete accepted gradient");
        for (column, &value) in direction.iter().enumerate() {
            let term = gradient[column] * value - compensation;
            let next = slope + term;
            compensation = (next - slope) - term;
            slope = next;
            for source in linearization.a.colptr[column]..linearization.a.colptr[column + 1] {
                activity[linearization.a.rowval[source]] += linearization.a.nzval[source] * value;
                self.evaluator.cooperation.completed(1).await?;
            }
            self.evaluator.cooperation.completed(1).await?;
        }
        let mut predicted_violation = 0.0;
        for (row, &value) in activity.iter().enumerate() {
            if !value.is_finite() {
                return Ok(Err(SearchStop::NoDescent));
            }
            predicted_violation += if row < linearization.equalities {
                value.abs()
            } else {
                value.max(0.0)
            };
            self.evaluator.cooperation.completed(1).await?;
        }
        let full_violation = current_violation;
        let full_predicted_violation = predicted_violation;
        let mut current_objective = point.f;
        if self.evaluator.uses_multiobjective_merit() {
            let Some((current, proposed)) = self
                .evaluator
                .merit_model(point, linearization, &activity, direction)
                .await?
            else {
                return Ok(Err(SearchStop::NoDescent));
            };
            slope = proposed.objective - current.objective;
            current_objective = current.objective;
            current_violation = current.violation;
            predicted_violation = proposed.violation;
        }
        for &value in multipliers {
            self.penalty = self.penalty.max(1.0 + value.abs());
            self.evaluator.cooperation.completed(1).await?;
        }
        let change = predicted_violation - current_violation;
        let mut predicted = slope + self.penalty * change;
        let mut feasibility_step = false;
        if predicted >= 0.0 && change < 0.0 {
            let needed = (slope.abs() + 1.0) / (-change);
            self.penalty = self.penalty.max(needed);
            predicted = slope + self.penalty * change;
        }
        if predicted >= 0.0
            && self.evaluator.uses_multiobjective_merit()
            && slope <= 0.0
            && full_predicted_violation < full_violation
        {
            // True-worst merit can be flat for a necessary gamma-only
            // correction. Preserve proposed gamma and require actual original
            // epigraph feasibility improvement without worsening true worst F.
            feasibility_step = true;
            predicted = full_predicted_violation - full_violation;
        }
        if !predicted.is_finite() || predicted >= 0.0 || !self.penalty.is_finite() {
            return Ok(Err(SearchStop::NoDescent));
        }
        let relative_active = self
            .evaluator
            .semi_infinite
            .as_ref()
            .is_some_and(|domain| domain.controls.line_active(self.iteration));
        let mut alpha = if let Some(domain) = &self.evaluator.semi_infinite {
            domain
                .controls
                .line_cap(
                    x,
                    direction,
                    self.iteration,
                    &self.evaluator.scaling,
                    self.evaluator.cooperation,
                )
                .await?
        } else {
            1.0
        };
        loop {
            let mut trial = workspace::capacity(x.len(), owner)?;
            let mut changed = false;
            let mut finite = true;
            let mut norm = 0.0f64;
            for variable in 0..x.len() {
                let value = (x[variable] + alpha * direction[variable])
                    .max(self.geometry.lower[variable])
                    .min(self.geometry.upper[variable]);
                finite &= value.is_finite();
                changed |= value != x[variable];
                if variable < self.evaluator.original_variables(x.len()) {
                    norm = norm.max((value - x[variable]).abs());
                }
                trial.push(value);
                self.evaluator.cooperation.completed(1).await?;
            }
            if !changed || alpha == 0.0 {
                return Ok(Err(if relative_active {
                    SearchStop::StepBound
                } else {
                    SearchStop::NoDescent
                }));
            }
            if let Some(domain) = &self.evaluator.semi_infinite {
                finite &= domain
                    .controls
                    .line_feasible(
                        x,
                        &trial,
                        self.iteration,
                        &self.evaluator.scaling,
                        self.evaluator.cooperation,
                    )
                    .await?;
            }
            if finite {
                let reused = self.evaluator.reuses_generation(x, &trial).await?;
                if !reused {
                    let cost = derivatives::sweep_cost(
                        self.evaluator,
                        &trial,
                        self.geometry.lower,
                        self.geometry.upper,
                        self.differences,
                    )
                    .await?;
                    if !self.evaluator.budget.permits_sweep_and_trial(cost) {
                        return Ok(Err(SearchStop::Limit));
                    }
                }
                let evaluated = if reused {
                    Ok(None)
                } else {
                    self.evaluator.evaluate(&trial).await?.map(Some)
                };
                match evaluated {
                    Err(EvaluationStop::Time) => return Ok(Err(SearchStop::Time)),
                    Err(EvaluationStop::BudgetExhausted) => return Ok(Err(SearchStop::Limit)),
                    Err(EvaluationStop::Undefined) => {}
                    Ok(mut candidate) => {
                        let borrowed = candidate.as_ref().unwrap_or(point);
                        if let Some(mut values) = self
                            .geometry
                            .constraint_values(&trial, borrowed, owner, self.evaluator.cooperation)
                            .await?
                        {
                            if values.roles != linearization.roles {
                                return Err(super::super::invalid(
                                    "trial changed original constraint row identities",
                                ));
                            }
                            let valid = !reused
                                || self
                                    .evaluator
                                    .stage_gamma(
                                        point,
                                        *trial.last().expect("attainment trial"),
                                        &mut values,
                                    )
                                    .await?;
                            if valid {
                                if let Some(terms) = self
                                    .evaluator
                                    .trial_merit(
                                        borrowed,
                                        *trial.last().expect("nonempty trial"),
                                        &values,
                                    )
                                    .await?
                                {
                                    let change = terms.objective - current_objective
                                        + self.penalty * (terms.violation - current_violation);
                                    let acceptable = if feasibility_step {
                                        let mut violation = 0.0;
                                        for (row, &rhs) in values.rhs.iter().enumerate() {
                                            violation += if row < values.equalities {
                                                rhs.abs()
                                            } else {
                                                (-rhs).max(0.0)
                                            };
                                            self.evaluator.cooperation.completed(1).await?;
                                        }
                                        violation.is_finite()
                                            && terms.objective <= current_objective
                                            && violation - full_violation
                                                <= 1e-4 * alpha * predicted
                                    } else {
                                        change.is_finite() && change <= 1e-4 * alpha * predicted
                                    };
                                    if acceptable {
                                        let complete = if let Some(candidate) = candidate.as_mut() {
                                            derivatives::complete(
                                                self.evaluator,
                                                &trial,
                                                self.geometry.lower,
                                                self.geometry.upper,
                                                candidate,
                                                self.differences,
                                            )
                                            .await?
                                        } else {
                                            Completion::Complete
                                        };
                                        match complete {
                                            Completion::Complete => {
                                                return Ok(Ok(Accepted {
                                                    x: trial,
                                                    point: candidate,
                                                    values,
                                                    alpha,
                                                    step_norm: norm,
                                                }))
                                            }
                                            Completion::Time => return Ok(Err(SearchStop::Time)),
                                            Completion::BudgetExhausted => {
                                                return Ok(Err(SearchStop::Limit))
                                            }
                                            Completion::Undefined => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            alpha *= 0.5;
            self.evaluator.cooperation.completed(1).await?;
        }
    }
}
