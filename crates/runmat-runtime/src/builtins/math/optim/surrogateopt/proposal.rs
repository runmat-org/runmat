//! Original-domain membership and repair before an expensive evaluation.
//! A numerical failure to produce a proposal is not an infeasibility proof.
use super::{history::History, invalid};
use crate::builtins::math::optim::{
    common::ScalarCooperation,
    intlinprog::feasibility::{Domain, Repair},
};
use crate::BuiltinResult;

pub(super) enum Prepared {
    Point(Vec<f64>),
    LinearInfeasible,
    TimeLimit,
    Unverified,
}

/// Repairs may change a point, so this accepts coordinates only, never known
/// objective/constraint values. Supplied initial measurements require an
/// unchanged, independently classified original point instead.
/// linear_tolerance is the original internal feasibility-owner tolerance,
/// not the public ConstraintTolerance used only for nonlinear observations.
pub(super) async fn prepare(
    mut point: Vec<f64>,
    domain: &mut Domain,
    linear_tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    cooperation.completed(0).await?;
    if !linear_tolerance.is_finite()
        || linear_tolerance < 0.0
        || point.len() != domain.bounds().0.len()
    {
        return Err(invalid("invalid original proposal geometry"));
    }
    for &coordinate in &point {
        if !coordinate.is_finite() {
            return Err(invalid("proposal contains a nonfinite coordinate"));
        }
        cooperation.completed(1).await?;
    }
    if !domain
        .classify(&point, linear_tolerance, cooperation)
        .await?
        .feasible
    {
        match domain.repair(&point, cooperation).await? {
            Repair::Unchanged => {}
            Repair::Point(repaired) => point = repaired,
            Repair::Infeasible => return Ok(Prepared::LinearInfeasible),
            Repair::TimeLimit => return Ok(Prepared::TimeLimit),
            Repair::Unverified => return Ok(Prepared::Unverified),
        }
        // The original model, not the proposal or relaxation geometry,
        // decides membership of the actual returned floating-point values.
        if !domain
            .classify(&point, linear_tolerance, cooperation)
            .await?
            .feasible
        {
            return Ok(Prepared::Unverified);
        }
    }
    Ok(Prepared::Point(point))
}

/// Construction may require distinct interpolation sites. Adaptive spacing is
/// separate: MinSampleDistance=0 must not silently become a positive/exact-site
/// exclusion rule, and repaired points require a fresh distance check.
pub(super) async fn already_evaluated(
    point: &[f64],
    history: &History,
    phase: super::queue::PhaseId,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for row in 0..history.len() {
        let trial = history.trial(super::history::TrialIndex(row));
        if trial.phase != phase {
            cooperation.completed(1).await?;
            continue;
        }
        let evaluated = &trial.point;
        if evaluated.variables() != point.len() {
            return Err(invalid("proposal/history original extent mismatch"));
        }
        let mut equal = true;
        for (column, &value) in point.iter().enumerate() {
            // +0 and -0 are the same numerical site. This does not permit
            // transferring supplied observations across a repaired point.
            equal &= value == evaluated.coordinate(column);
            cooperation.completed(1).await?;
            if !equal {
                break;
            }
        }
        if equal {
            return Ok(true);
        }
    }
    Ok(false)
}
