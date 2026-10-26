//! One typed active result: proven optimum/infeasibility or evaluated-only stop.
use super::super::model::{ActiveTermination as Reason, Method};
use super::*;
pub(super) enum State {
    Optimal {
        step: working_set::Step,
        metrics: optimality::Metrics,
    },
    Stopped {
        point: Vec<f64>,
        reason: Reason,
    },
    Infeasible,
}
pub(super) struct Prepared<'a, V> {
    pub active: Option<Vec<bool>>,
    pub model: Model,
    pub controls: options::Controls,
    pub products: compiled::Products,
    pub state: State,
    pub validation: &'a mut V,
    pub c: &'a mut ScalarCooperation,
}
fn status(reason: Reason) -> (i32, &'static str) {
    match reason {
        Reason::Optimal=>(1,"Original residual and active-set KKT tolerances satisfied."),
        Reason::IterationLimit=>(0,"Iteration limit reached; returning the actual stopped candidate."),
        Reason::ObjectiveLimit=>(-3,"Feasible objective is below ObjectiveLimit; this is a threshold stop, not an unboundedness certificate."),
        Reason::IllConditioned=>(-4,"Ill-conditioning or StepTolerance prevents further optimization."),
        Reason::NoDirection=>(-8,"Unable to compute a qualified active-set step."),
        Reason::Unverified=>(-8,"Candidate could not be authenticated against the original model."),
        Reason::Infeasible=>(-2,"Original constraints have an exact infeasibility certificate."),
    }
}
pub(super) async fn publish<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
) -> BuiltinResult<Outcome> {
    let Prepared {
        active,
        model,
        controls,
        products,
        state,
        validation,
        c,
    } = prepared;
    let (mut point, mut reason, mut duals, metrics) = match state {
        State::Infeasible => {
            let (flag, message) = status(Reason::Infeasible);
            let mut out = Outcome::failure(flag, controls.iterations(), message);
            out.method = Method::ActiveSet {
                termination: Reason::Infeasible,
            };
            controls.summary(&out, c.runtime())?;
            return Ok(out);
        }
        State::Optimal { step, metrics } => (
            step.direction,
            Reason::Optimal,
            Some(step.multipliers),
            Some(metrics),
        ),
        State::Stopped { point, reason } => (point, reason, None, None),
    };
    if reason == Reason::Optimal
        && !validation
            .solution(
                &point,
                compiled::Rows::original(&model.rows.roles),
                duals.as_ref().expect("optimal working multipliers"),
                controls.feasibility,
                controls.optimality,
                c,
            )
            .await?
    {
        reason = Reason::Unverified;
        duals = None;
    }
    if reason != Reason::Optimal && !validation.stopped_candidate(&point, c).await? {
        reason = Reason::Unverified;
        point.clear();
    }
    let numerical = if point.is_empty() {
        None
    } else {
        super::super::residual::residual(model.objective(), &point, allocation, c).await?
    };
    if reason == Reason::Optimal && numerical.is_none() {
        reason = Reason::Unverified;
        duals = None;
    }
    let mut lambda = None;
    if reason == Reason::Optimal && products == compiled::Products::WithMultipliers {
        let mut lower = zeros(model.c.n, c).await?;
        let mut upper = zeros(model.c.n, c).await?;
        let mut eqlin = zeros(model.rows.equalities, c).await?;
        let mut ineqlin = zeros(model.rows.inequalities, c).await?;
        for (&role, dual) in model.rows.roles.iter().zip(duals.expect("verified duals")) {
            match role {
                conic::LinearRowRole::Equality(i) => eqlin[i] = dual,
                conic::LinearRowRole::Inequality(i) => ineqlin[i] = dual,
                conic::LinearRowRole::Lower(i) => lower[i] = dual,
                conic::LinearRowRole::Upper(i) => upper[i] = dual,
            }
            c.completed(1).await?;
        }
        lambda = Some(super::super::model::Lambda {
            lower,
            upper,
            eqlin,
            ineqlin,
        });
    }
    let mut violation = metrics
        .as_ref()
        .filter(|_| reason == Reason::Optimal)
        .map_or(f64::NAN, |value| value.violation);
    if reason != Reason::Optimal && !point.is_empty() {
        if let Some(activity) = working_set::rows::product(
            &model.rows.matrix,
            &point,
            runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
            c,
        )
        .await?
        {
            let mut maximum = 0f64;
            for (row, (&value, &rhs)) in activity.iter().zip(&model.rows.rhs).enumerate() {
                let error = value - rhs;
                maximum = maximum.max(if row < model.rows.equalities {
                    error.abs()
                } else {
                    error.max(0.)
                });
                c.completed(1).await?;
            }
            if maximum.is_finite() {
                violation = maximum;
            }
        }
    }
    let (flag, message) = status(reason);
    let (residual, resnorm) = match numerical {
        Some(value) => (Some(value.values), Some(value.value)),
        None => (None, None),
    };
    let out = Outcome {
        method: Method::ActiveSet {
            termination: reason,
        },
        x: (!point.is_empty()).then_some(point),
        residual,
        resnorm,
        lambda,
        flag,
        iterations: controls.iterations(),
        violation,
        optimality: metrics
            .filter(|_| reason == Reason::Optimal)
            .map_or(f64::NAN, |m| m.optimality),
        gap: f64::NAN,
        message,
    };
    if reason != Reason::Unverified {
        if let Some(active) = active {
            validation.retain_working_set(
                working_set::warm::retain(
                    working_set::warm::StoredRows {
                        matrix: model.rows.matrix,
                        rhs: model.rows.rhs,
                        roles: model.rows.roles,
                        equalities: model.rows.equalities,
                    },
                    active,
                    warm_policy(),
                    c,
                )
                .await?,
            );
        }
    }
    controls.summary(&out, c.runtime())?;
    Ok(out)
}
