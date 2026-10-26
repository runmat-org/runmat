//! Full original-row KKT remains the success authority, not a scaled or
//! projected inner residual or a quadratic-kernel status.
use super::super::{model::Role, validation};
use super::*;
#[derive(Clone, Copy)]
pub(super) enum State {
    Optimal,
    Continue,
    Numerical,
}
pub(super) async fn refresh(
    out: &mut Outcome,
    geometry: &Geometry,
    equality: Option<&equality::Equality<'_>>,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<State> {
    let point = out.point.as_ref().expect("defined reflective point");
    let Some(values) = geometry
        .constraint_values(&out.x, point, IDENTITY, evaluator.cooperation)
        .await?
    else {
        out.constraints = None;
        out.multipliers_available = false;
        out.optimality = None;
        out.violation = None;
        return Ok(State::Numerical);
    };
    let constraints = if let Some(mut cached) = out.constraints.take() {
        // Admitted reflective geometry is immutable and contains no dynamic
        // nonlinear rows. Retain its actual CSC, replacing only this point's
        // original residual generation after checking every row identity.
        if cached.semi_infinite.is_some()
            || cached.equalities != values.equalities
            || cached.roles.len() != values.roles.len()
        {
            return Err(super::super::invalid(
                "reflective immutable geometry changed",
            ));
        }
        for (&before, &after) in cached.roles.iter().zip(&values.roles) {
            if before != after {
                return Err(super::super::invalid(
                    "reflective immutable row identity changed",
                ));
            }
            evaluator.cooperation.completed(1).await?;
        }
        cached.b = values.rhs;
        cached
    } else {
        geometry
            .linearize_with_values(&out.x, point, values, IDENTITY, evaluator.cooperation)
            .await?
    };
    let gradient = point
        .gradient
        .as_ref()
        .expect("supplied reflective gradient");
    let multipliers = if let Some(equality) = equality {
        let Some(values) = equality
            .multipliers(gradient, evaluator.cooperation)
            .await?
        else {
            out.constraints = Some(constraints);
            out.multipliers_available = false;
            out.optimality = None;
            out.violation = None;
            return Ok(State::Numerical);
        };
        if values.len() != constraints.roles.len() {
            return Err(super::super::invalid("equality multiplier roles changed"));
        }
        values
    } else {
        let mut values = workspace::capacity(constraints.roles.len(), IDENTITY)?;
        for (row, &role) in constraints.roles.iter().enumerate() {
            values.push(match role {
                Role::Lower(index) if constraints.b[row] <= parameters.feasibility => {
                    gradient[index].max(0.0)
                }
                Role::Upper(index) if constraints.b[row] <= parameters.feasibility => {
                    (-gradient[index]).max(0.0)
                }
                Role::Lower(_) | Role::Upper(_) => 0.0,
                _ => {
                    return Err(super::super::invalid(
                        "bound-reflective model contains an unsupported constraint role",
                    ))
                }
            });
            evaluator.cooperation.completed(1).await?;
        }
        values
    };
    let checked = validation::check(
        gradient,
        &constraints,
        &multipliers,
        parameters.feasibility,
        parameters.optimality,
        IDENTITY,
        evaluator.cooperation,
    )
    .await?;
    out.constraints = Some(constraints);
    out.multipliers = multipliers;
    match checked {
        Some(checked) => {
            out.violation = Some(checked.violation);
            out.optimality = Some(checked.optimality);
            out.multipliers_available = true;
            Ok(if checked.valid {
                State::Optimal
            } else {
                State::Continue
            })
        }
        None => {
            out.multipliers_available = false;
            out.optimality = None;
            out.violation = None;
            Ok(State::Numerical)
        }
    }
}
