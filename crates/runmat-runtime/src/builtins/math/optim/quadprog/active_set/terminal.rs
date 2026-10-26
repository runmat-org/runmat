//! Original evidence, typed stops, immutable warm-state publication.
use super::*;
pub(super) enum State {
    Optimal {
        step: working_set::Step,
        checked: validation::Checked,
    },
    Stopped {
        point: Vec<f64>,
        reason: Reason,
    },
    Ray {
        point: Vec<f64>,
        direction: Vec<f64>,
    },
    Failure(Reason),
}
pub(super) struct Prepared<'a, V> {
    pub model: Model,
    pub controls: options::Controls,
    pub products: compiled::Products,
    pub active: Option<Vec<bool>>,
    pub state: State,
    pub validation: &'a mut V,
    pub c: &'a mut ScalarCooperation,
}
fn status(reason: Reason) -> (i32, &'static str) {
    match reason {
        Reason::Optimal => (1, "Original quadratic and active-set KKT checks satisfied."),
        Reason::IterationLimit => (0, "Iteration limit reached; returning the current candidate."),
        Reason::TimeLimit => (0, "MaxTime reached; returning the current candidate."),
        Reason::ObjectiveLimit => (-3, "Feasible objective is below ObjectiveLimit; threshold stop, not an unboundedness certificate."),
        Reason::StepStagnation => (-8, "StepTolerance prevents a qualified active-set step."),
        Reason::Numerical => (-8, "Unable to compute a qualified active-set direction."),
        Reason::Unverified => (-8, "Candidate failed original-model checks."),
        Reason::Infeasible => (-2, "Original constraints have an exact infeasibility certificate."),
        Reason::Unbounded => (-3, "Exact feasible anchor and descending flat recession direction verified."),
        Reason::Nonconvex => (-6, "H is not positive semidefinite on the original equality null space."),
    }
}
pub(super) async fn publish<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
) -> BuiltinResult<Outcome> {
    let Prepared {
        model,
        controls,
        products,
        active,
        state,
        validation,
        c,
    } = prepared;
    let (mut reason, point, duals, measurements) = match state {
        State::Failure(reason) => (reason, None, None, None),
        State::Optimal { step, checked } => (
            Reason::Optimal,
            Some(step.direction),
            Some(step.multipliers),
            Some(checked),
        ),
        State::Stopped { point, reason } => (reason, Some(point), None, None),
        State::Ray { point, direction } => {
            let reason = if proof::certify(&model, &point, &direction, validation, c).await? {
                Reason::Unbounded
            } else {
                Reason::Unverified
            };
            (reason, None, None, None)
        }
    };
    let mut out = Outcome::failure(status(reason).0, controls.iterations(), status(reason).1);
    if let Some(point) = point {
        let checked = match measurements {
            Some(value) => Some(value),
            None => validation::primal(&model, &point, c).await?,
        };
        let accepted = if reason == Reason::Optimal {
            validation
                .solution(
                    &point,
                    &model.roles,
                    duals.as_deref().expect("optimal duals"),
                    controls.feasibility,
                    controls.optimality,
                    c,
                )
                .await?
        } else {
            validation.stopped_candidate(&point, c).await?
        };
        if !accepted || checked.is_none() {
            reason = Reason::Unverified;
        }
        if reason != Reason::Unverified {
            let checked = checked.expect("measured original quadratic");
            out.fval = Some(checked.objective);
            out.violation = checked.violation;
            out.optimality = checked.optimality;
            out.gap = checked.gap;
            out.x = Some(point);
            if reason == Reason::Optimal && products == compiled::Products::WithMultipliers {
                out.lambda = Some(
                    model
                        .lambda(duals.as_deref().expect("checked optimal duals"), c)
                        .await?,
                );
            }
        }
    }
    if reason != Reason::Unverified {
        if let Some(active) = active {
            validation.retain_working_set(
                working_set::warm::retain(
                    working_set::warm::StoredRows {
                        matrix: model.a,
                        rhs: model.b,
                        roles: model.roles,
                        equalities: model.eq,
                    },
                    active,
                    curvature::POLICY,
                    c,
                )
                .await?,
            );
        }
    }
    (out.flag, out.message) = status(reason);
    out.method = model::Method::ActiveSet(reason);
    controls.summary(&out, c.runtime())?;
    Ok(out)
}
