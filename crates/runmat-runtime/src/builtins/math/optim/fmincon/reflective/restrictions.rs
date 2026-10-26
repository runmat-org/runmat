//! Public reflective geometry is narrower than general constrained geometry.
//! No nonlinear callback or trial model is needed to establish these roles.
use super::super::{invalid, model::Geometry};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim::fmincon) enum Mode {
    Bounds,
    Equalities,
}

pub(in crate::builtins::math::optim::fmincon) fn select(
    geometry: &Geometry,
    nonlinear: bool,
    supplied_gradient: bool,
) -> BuiltinResult<Mode> {
    if !supplied_gradient {
        return Err(invalid("trust-region-reflective requires SpecifyObjectiveGradient=true and an objective gradient"));
    }
    if nonlinear || geometry.inequality.matrix.m != 0 {
        return Err(invalid("trust-region-reflective supports only bounds or linear equalities, not nonlinear or linear inequality constraints"));
    }
    let bounds = geometry
        .lower
        .iter()
        .zip(&geometry.upper)
        .any(|(&lower, &upper)| lower.is_finite() || upper.is_finite());
    if bounds && geometry.equality.matrix.m != 0 {
        return Err(invalid(
            "trust-region-reflective cannot combine bounds with linear equalities",
        ));
    }
    if geometry
        .lower
        .iter()
        .zip(&geometry.upper)
        .any(|(&lower, &upper)| lower >= upper)
    {
        return Err(invalid(
            "trust-region-reflective does not permit equal or inconsistent lower and upper bounds",
        ));
    }
    Ok(if geometry.equality.matrix.m == 0 {
        Mode::Bounds
    } else {
        Mode::Equalities
    })
}
