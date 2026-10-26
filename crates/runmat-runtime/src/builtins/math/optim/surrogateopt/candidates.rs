//! Actual cheap proposal workspace: physical points only, no normalized
//! mirror, trial copies, objective observations or unverified repair outputs.
use super::{
    domain::{BoxGeometry, BoxView},
    invalid,
    merit::{CandidateIndex, Proposals},
    proposal::{self, Prepared},
};
use crate::{
    builtins::math::optim::{common::ScalarCooperation, intlinprog::feasibility::Domain},
    BuiltinResult,
};

pub(super) enum Added {
    Point,
    LinearInfeasible,
    TimeLimit,
    Unverified,
}
pub(super) struct Builder<'a> {
    domain: &'a mut Domain,
    geometry: &'a BoxGeometry,
    points: Vec<Vec<f64>>,
    limit: usize,
    linear_tolerance: f64,
}
pub(super) struct Candidates<'a> {
    domain: BoxView<'a>,
    points: Vec<Vec<f64>>,
}
impl<'a> Builder<'a> {
    pub(super) fn new(
        domain: &'a mut Domain,
        geometry: &'a BoxGeometry,
        limit: usize,
        linear_tolerance: f64,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<Self> {
        geometry.view(domain)?;
        if limit == 0 || !linear_tolerance.is_finite() || linear_tolerance < 0.0 {
            return Err(invalid("invalid modeled candidate workspace controls"));
        }
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            super::NAME,
            "finite host-memory execution has no complete candidate workspace bound",
        )?;
        let points = runmat_value::exact_capacity_vec(limit).map_err(|source| {
            crate::build_runtime_error("surrogateopt: modeled candidate metadata allocation failed")
                .with_builtin(super::NAME)
                .with_source(source)
                .build()
        })?;
        Ok(Self {
            domain,
            geometry,
            points,
            limit,
            linear_tolerance,
        })
    }
    pub(super) fn domain(&self) -> BuiltinResult<BoxView<'_>> {
        self.geometry.view(self.domain)
    }
    pub(super) fn len(&self) -> usize {
        self.points.len()
    }
    pub(super) async fn push(
        &mut self,
        point: Vec<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Added> {
        if self.len() == self.limit {
            return Err(invalid("modeled candidate workspace is full"));
        }
        match proposal::prepare(point, self.domain, self.linear_tolerance, cooperation).await? {
            Prepared::Point(point) => {
                let view = self.geometry.view(self.domain)?;
                for axis in 0..view.dimensions() {
                    let value = view.unit_coordinate(&point, axis);
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Ok(Added::Unverified);
                    }
                    cooperation.completed(1).await?;
                }
                self.points.push(point);
                Ok(Added::Point)
            }
            Prepared::LinearInfeasible => Ok(Added::LinearInfeasible),
            Prepared::TimeLimit => Ok(Added::TimeLimit),
            Prepared::Unverified => Ok(Added::Unverified),
        }
    }
    pub(super) fn finish(self) -> BuiltinResult<Candidates<'a>> {
        Ok(Candidates {
            domain: self.geometry.view(self.domain)?,
            points: self.points,
        })
    }
}
impl Candidates<'_> {
    pub(super) fn domain(&self) -> &BoxView<'_> {
        &self.domain
    }
    /// Transfer only the selected physical point. No modeled F or constraint
    /// prediction is ever attached to an expensive objective submission.
    pub(super) fn take(mut self, index: CandidateIndex) -> BuiltinResult<Vec<f64>> {
        if index.0 >= self.points.len() {
            return Err(invalid("selected candidate index is out of range"));
        }
        Ok(self.points.swap_remove(index.0))
    }
}
impl Proposals for Candidates<'_> {
    fn len(&self) -> usize {
        self.points.len()
    }
    fn dimensions(&self) -> usize {
        self.domain.dimensions()
    }
    fn coordinate(&self, row: usize, axis: usize) -> f64 {
        self.domain.unit_coordinate(&self.points[row], axis)
    }
}
