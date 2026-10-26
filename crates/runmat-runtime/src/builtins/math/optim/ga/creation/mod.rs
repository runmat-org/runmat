//! Initial generation creation and original-domain admission.
mod custom;
mod domain;
pub(super) mod factory;
mod geometry_wire;
mod initial;
mod integer;
mod nonlinear;
mod numeric;
mod parallel;
mod sobol;
#[cfg(test)]
mod tests;
pub(crate) mod worker;
pub(super) use initial::create;
pub(super) use initial::create_with;
pub(super) use initial::Initial;
pub(super) use integer::fill as fill_uniform_integer;
pub(super) use numeric::Draft;
pub(super) use sobol::fill as fill_sobol;

use super::{common, ScalarCooperation};
use crate::BuiltinResult;
use common::{global_time::TimeLimit, solver_timing::SolverTiming};

pub(super) enum Admission {
    Complete,
    Infeasible,
    TimeLimit,
    Unverified,
}
pub(super) struct ChangedRows {
    prefix: usize,
    rows: Vec<usize>,
}
impl ChangedRows {
    fn new(prefix: usize, cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        Ok(Self {
            prefix,
            rows: super::POLICY.capacity(prefix, cooperation.runtime())?,
        })
    }
    fn record(&mut self, row: usize) {
        if row < self.prefix {
            self.rows.push(row);
        }
    }
}

/// Every numeric initial row, including a supplied row, enters the same
/// original-domain owner. A failed repair is never installed as a feasible
/// seed, and repair work does not consume a fitness observation.
pub(super) async fn admit(
    draft: &mut Draft,
    domain: &mut impl super::linear::Geometry,
    timing: &SolverTiming,
    maximum: TimeLimit,
    changed: &mut ChangedRows,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Admission> {
    use super::super::intlinprog::feasibility::Repair;
    for row in 0..draft.members() {
        if maximum.reached(timing.elapsed()) {
            return Ok(Admission::TimeLimit);
        }
        let point = draft.row(row, cooperation).await?;
        match domain.repair(&point, cooperation).await? {
            Repair::Unchanged => (),
            Repair::Point(point) => {
                if draft.set(row, &point, false, cooperation).await? {
                    changed.record(row);
                }
            }
            Repair::Infeasible => return Ok(Admission::Infeasible),
            Repair::TimeLimit => return Ok(Admission::TimeLimit),
            Repair::Unverified => return Ok(Admission::Unverified),
        }
        cooperation.completed(1).await?;
    }
    Ok(Admission::Complete)
}
