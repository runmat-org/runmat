//! Positive primal-dual trials use the actual nonlinear logarithmic-barrier
//! merit. Rejected values never trigger derivative sweeps or exception replay.
use super::super::{
    derivatives::{self, Completion},
    evaluation::{EvaluationStop, Evaluator, Point},
    model::{ConstraintValues, Geometry, Linearization},
    outcome::Parameters,
    workspace,
};
use super::{
    newton::Direction,
    state::{Layout, Row, State},
};
use crate::BuiltinResult;
pub(super) struct EvaluatedTrial {
    pub x: Vec<f64>,
    pub point: Point,
    pub constraints: ConstraintValues,
}
pub(super) type Generation = Option<EvaluatedTrial>;
pub(super) struct Accepted {
    pub generation: Generation,
    pub state: State,
    pub alpha: f64,
    pub step: f64,
    pub state_step: f64,
}
pub(super) type Result = std::result::Result<Accepted, SearchStop>;
pub(super) enum SearchStop {
    Time,
    Limit,
    NoDescent,
}

pub(super) async fn feasibility(
    rhs: &[f64],
    layout: &Layout,
    state: &State,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    if rhs.len() != layout.rows.len() {
        return Ok(None);
    }
    let mut value = 0.0;
    for (&rhs, &kind) in rhs.iter().zip(&layout.rows) {
        value += match kind {
            Row::Equality(_) => rhs.abs(),
            Row::Inequality(index) => (state.s[index] - rhs).abs(),
            Row::Fixed => 0.0,
        };
        if !value.is_finite() {
            return Ok(None);
        }
        cooperation.completed(1).await?;
    }
    Ok(Some(value))
}
async fn merit(
    f: f64,
    rhs: &[f64],
    layout: &Layout,
    state: &State,
    penalty: f64,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    let Some(residual) = feasibility(rhs, layout, state, cooperation).await? else {
        return Ok(None);
    };
    let mut value = f + penalty * residual;
    if !value.is_finite() {
        return Ok(None);
    }
    for &slack in &state.s {
        if slack <= 0.0 || !slack.is_finite() {
            return Ok(None);
        }
        value -= state.mu * slack.ln();
        if !value.is_finite() {
            return Ok(None);
        }
        cooperation.completed(1).await?;
    }
    Ok(Some(value))
}
async fn positive_trial(
    state: &State,
    direction: &Direction,
    alpha: f64,
    owner: runmat_builtins::BuiltinCatalogIdentity,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Option<(State, f64)>> {
    let mut s = workspace::capacity(state.s.len(), owner)?;
    let mut z = workspace::capacity(state.z.len(), owner)?;
    let mut y = workspace::capacity(state.y.len(), owner)?;
    let mut change = 0.0f64;
    for index in 0..state.s.len() {
        let slack = state.s[index] + alpha * direction.s[index];
        let dual = state.z[index] + alpha * direction.z[index];
        if !slack.is_finite() || !dual.is_finite() || slack <= 0.0 || dual <= 0.0 {
            return Ok(None);
        }
        change = change
            .max(((slack - state.s[index]) / (1.0 + state.s[index])).abs())
            .max(((dual - state.z[index]) / (1.0 + state.z[index])).abs());
        s.push(slack);
        z.push(dual);
        cooperation.completed(1).await?;
    }
    for (&current, &delta) in state.y.iter().zip(&direction.y) {
        let value = current + alpha * delta;
        if !value.is_finite() {
            return Ok(None);
        }
        change = change.max(((value - current) / (1.0 + current.abs())).abs());
        y.push(value);
        cooperation.completed(1).await?;
    }
    Ok(Some((
        State {
            s,
            z,
            y,
            mu: state.mu,
        },
        change,
    )))
}

pub(super) async fn search(
    x: &[f64],
    point: &Point,
    original: &Linearization,
    geometry: &Geometry,
    layout: &Layout,
    state: &State,
    direction: &Direction,
    parameters: &Parameters<'_>,
    penalty: &mut f64,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Result> {
    let owner = evaluator.policy.owner;
    let Some(residual) = feasibility(&original.b, layout, state, evaluator.cooperation).await?
    else {
        return Ok(Err(SearchStop::NoDescent));
    };
    let mut slope = 0.0;
    let gradient = point.gradient.as_ref().expect("complete barrier gradient");
    for (&gradient, &delta) in gradient.iter().zip(&direction.x) {
        slope += gradient * delta;
        if !slope.is_finite() {
            return Ok(Err(SearchStop::NoDescent));
        }
        evaluator.cooperation.completed(1).await?;
    }
    for (&slack, &delta) in state.s.iter().zip(&direction.s) {
        slope -= state.mu * (delta / slack);
        if !slope.is_finite() {
            return Ok(Err(SearchStop::NoDescent));
        }
        evaluator.cooperation.completed(1).await?;
    }
    for &value in state.z.iter().chain(&state.y) {
        *penalty = penalty.max(value.abs() + 1.0);
        evaluator.cooperation.completed(1).await?;
    }
    let mut predicted = slope - *penalty * residual;
    if predicted >= 0.0 && residual > 0.0 {
        *penalty = penalty.max((slope.abs() + 1.0) / residual);
        predicted = slope - *penalty * residual;
    }
    if !predicted.is_finite() || predicted >= 0.0 || !penalty.is_finite() {
        return Ok(Err(SearchStop::NoDescent));
    }
    let Some(current) = merit(
        point.f,
        &original.b,
        layout,
        state,
        *penalty,
        evaluator.cooperation,
    )
    .await?
    else {
        return Ok(Err(SearchStop::NoDescent));
    };
    let mut alpha = 1.0f64;
    for (&value, &delta) in state
        .s
        .iter()
        .zip(&direction.s)
        .chain(state.z.iter().zip(&direction.z))
    {
        if delta < 0.0 {
            alpha = alpha.min(-0.995 * value / delta);
        }
        evaluator.cooperation.completed(1).await?;
    }
    for index in 0..x.len() {
        if geometry.lower[index] != geometry.upper[index] {
            let delta = direction.x[index];
            if delta > 0.0 && geometry.upper[index].is_finite() {
                alpha = alpha.min(0.995 * ((geometry.upper[index] - x[index]) / delta));
            }
            if delta < 0.0 && geometry.lower[index].is_finite() {
                alpha = alpha.min(0.995 * ((geometry.lower[index] - x[index]) / delta));
            }
        }
        evaluator.cooperation.completed(1).await?;
    }
    loop {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Ok(Err(SearchStop::NoDescent));
        }
        let mut trial = workspace::capacity(x.len(), owner)?;
        let mut changed = false;
        let mut step = 0.0f64;
        let mut finite = true;
        for index in 0..x.len() {
            let lower = geometry.lower[index];
            let upper = geometry.upper[index];
            let mut value = x[index] + alpha * direction.x[index];
            if lower == upper {
                value = lower;
            } else {
                if value <= lower {
                    value = lower.next_up();
                }
                if value >= upper {
                    value = upper.next_down();
                }
                finite &= value > lower && value < upper;
            }
            finite &= value.is_finite();
            changed |= value != x[index];
            step = step.max((value - x[index]).abs());
            trial.push(value);
            evaluator.cooperation.completed(1).await?;
        }
        if finite {
            if let Some((trial_state, state_step)) =
                positive_trial(state, direction, alpha, owner, evaluator.cooperation).await?
            {
                if !changed {
                    if state_step == 0.0 {
                        return Ok(Err(SearchStop::NoDescent));
                    }
                    if merit(
                        point.f,
                        &original.b,
                        layout,
                        &trial_state,
                        *penalty,
                        evaluator.cooperation,
                    )
                    .await?
                    .is_some_and(|value| value <= current + 1e-4 * alpha * predicted)
                    {
                        return Ok(Ok(Accepted {
                            generation: None,
                            state: trial_state,
                            alpha,
                            step,
                            state_step,
                        }));
                    }
                } else {
                    let cost = derivatives::sweep_cost(
                        evaluator,
                        &trial,
                        &geometry.lower,
                        &geometry.upper,
                        parameters.differences,
                    )
                    .await?;
                    if !evaluator.budget.permits_sweep_and_trial(cost) {
                        return Ok(Err(SearchStop::Limit));
                    }
                    match evaluator.evaluate(&trial).await? {
                        Err(EvaluationStop::Time) => return Ok(Err(SearchStop::Time)),
                        Err(EvaluationStop::BudgetExhausted) => return Ok(Err(SearchStop::Limit)),
                        Err(EvaluationStop::Undefined) => {}
                        Ok(mut next) => {
                            let Some(values) = geometry
                                .constraint_values(&trial, &next, owner, evaluator.cooperation)
                                .await?
                            else {
                                // Only numerical original-model overflow rejects this trial.
                                // Callback, cancellation and allocation errors propagate above.
                                alpha *= 0.5;
                                evaluator.cooperation.completed(1).await?;
                                continue;
                            };
                            if values.roles != original.roles {
                                return Err(super::super::invalid(
                                    "nonlinear trial changed original row identities",
                                ));
                            }
                            if merit(
                                next.f,
                                &values.rhs,
                                layout,
                                &trial_state,
                                *penalty,
                                evaluator.cooperation,
                            )
                            .await?
                            .is_some_and(|value| value <= current + 1e-4 * alpha * predicted)
                            {
                                match derivatives::complete(
                                    evaluator,
                                    &trial,
                                    &geometry.lower,
                                    &geometry.upper,
                                    &mut next,
                                    parameters.differences,
                                )
                                .await?
                                {
                                    Completion::Time => return Ok(Err(SearchStop::Time)),
                                    Completion::BudgetExhausted => {
                                        return Ok(Err(SearchStop::Limit))
                                    }
                                    Completion::Undefined => {}
                                    Completion::Complete => {
                                        return Ok(Ok(Accepted {
                                            generation: Some(EvaluatedTrial {
                                                x: trial,
                                                point: next,
                                                constraints: values,
                                            }),
                                            state: trial_state,
                                            alpha,
                                            step,
                                            state_step,
                                        }))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        alpha *= 0.5;
        evaluator.cooperation.completed(1).await?;
    }
}
