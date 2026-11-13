//! Goal-attainment globalization uses the worst normalized objective and hard
//! original constraints, rather than pretending the smooth gamma merit is the
//! documented multiobjective merit. Generated nonzero-magnitude rows are not
//! counted twice as hard constraints.
use super::super::{
    evaluation::Point,
    model::{Linearization, Role},
};
use crate::builtins::math::optim::common::{
    multiobjective::{formulation::Formulation, invalid},
    ScalarCooperation,
};
use crate::BuiltinResult;

pub(in crate::builtins::math::optim::fmincon) struct Terms {
    pub objective: f64,
    pub violation: f64,
}
fn generated_weight(formulation: &Formulation, role: Role) -> Option<f64> {
    match role {
        Role::InequalityNonlinear(row) => formulation.rows.get(row).map(|row| row.weight),
        _ => None,
    }
}

pub(in crate::builtins::math::optim::fmincon) async fn current(
    formulation: &Formulation,
    point: &Point,
    rhs: &[f64],
    roles: &[Role],
    equalities: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Terms>> {
    let payload = point
        .objectives
        .as_ref()
        .expect("typed vector objective payload");
    let Some(objective) = formulation.worst(&payload.values, cooperation).await? else {
        // With no nonzero weights gamma is unconstrained, not a fabricated
        // finite minimizer of the hard-constraint feasibility problem.
        return Ok(None);
    };
    if roles.len() != rhs.len() {
        return Err(invalid(
            formulation.kind,
            "original constraint row identities are inconsistent",
        ));
    }
    let mut violation = 0.0;
    for (row, (&rhs, &role)) in rhs.iter().zip(roles).enumerate() {
        if !generated_weight(formulation, role).is_some_and(|weight| weight > 0.0) {
            violation += if row < equalities {
                rhs.abs()
            } else {
                (-rhs).max(0.0)
            };
            if !violation.is_finite() {
                return Ok(None);
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(Some(Terms {
        objective,
        violation,
    }))
}

pub(in crate::builtins::math::optim::fmincon) async fn predicted(
    formulation: &Formulation,
    point: &Point,
    constraints: &Linearization,
    activity: &[f64],
    direction: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Terms>> {
    if activity.len() != constraints.roles.len() || activity.len() != constraints.b.len() {
        return Err(invalid(
            formulation.kind,
            "linearized constraint row identities are inconsistent",
        ));
    }
    let Some(&gamma_step) = direction.last() else {
        return Err(invalid(formulation.kind, "missing attainment direction"));
    };
    let mut worst: Option<f64> = None;
    let mut violation = 0.0;
    for (row, (&value, &role)) in activity.iter().zip(&constraints.roles).enumerate() {
        if !value.is_finite() {
            return Ok(None);
        }
        match generated_weight(formulation, role) {
            Some(weight) if weight > 0.0 => {
                // activity = sign*(F-goal)-magnitude*gamma
                //          + (sign*JF,-magnitude)*direction.
                // Removing gamma recovers the true linearized normalized F.
                let objective = value / weight + point.f + gamma_step;
                if !objective.is_finite() {
                    return Ok(None);
                }
                worst = Some(worst.map_or(objective, |old| old.max(objective)));
            }
            _ => {
                violation += if row < constraints.equalities {
                    value.abs()
                } else {
                    value.max(0.0)
                };
                if !violation.is_finite() {
                    return Ok(None);
                }
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(worst.map(|objective| Terms {
        objective,
        violation,
    }))
}
