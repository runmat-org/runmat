//! Genetic variation borrows the original Pattern rows and bound owner.
//! Repairs use Pattern's existing phase-one kernel; no CSC/model is cloned.
use super::super::super::{
    buffer, invalid, linear::Linear, phase_one, timing::AuxiliaryTiming, IDENTITY,
};
use super::super::Bounds;
use crate::builtins::math::optim::{
    common::{global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation},
    ga::linear::{linear_tolerance, Geometry as GeneticGeometry},
    intlinprog::feasibility::{Membership, Repair},
};
use crate::BuiltinResult;

pub(super) struct Geometry<'a> {
    linear: &'a Linear,
    bounds: &'a Bounds,
    timing: &'a AuxiliaryTiming<'a>,
    maximum: TimeLimit,
    parent_tolerance: f64,
    tolerance: f64,
}
impl<'a> Geometry<'a> {
    pub fn new(
        linear: &'a Linear,
        bounds: &'a Bounds,
        timing: &'a AuxiliaryTiming<'a>,
        maximum: TimeLimit,
        parent_tolerance: f64,
        tolerance: f64,
    ) -> BuiltinResult<Self> {
        let (lower, upper) = bounds.original();
        if lower.len() != linear.matrix.n
            || upper.len() != linear.matrix.n
            || !parent_tolerance.is_finite()
            || parent_tolerance < 0.
            || !tolerance.is_finite()
            || tolerance < 0.
        {
            return Err(invalid("invalid borrowed genetic domain geometry"));
        }
        Ok(Self {
            linear,
            bounds,
            timing,
            maximum,
            parent_tolerance,
            tolerance,
        })
    }
    fn expired(&self) -> bool {
        self.timing.reached(self.maximum)
    }
    fn effective_tolerance(&self, local: f64) -> f64 {
        self.parent_tolerance.min(linear_tolerance(local))
    }
}
impl GeneticGeometry for Geometry<'_> {
    fn bounds(&self) -> (&[f64], &[f64]) {
        self.bounds.original()
    }
    fn integers(&self) -> &[usize] {
        &[]
    }
    fn rhs(&self) -> &[f64] {
        &self.linear.rhs
    }
    fn equalities(&self) -> usize {
        self.linear.equalities
    }
    fn repair_nodes(&self) -> usize {
        0
    } // No branch-and-bound node is run here.
    fn time_limit_reached(&self, _: &SolverTiming, maximum: TimeLimit) -> bool {
        self.timing.reached(maximum)
    }
    fn retune(&mut self, maximum: TimeLimit, tolerance: f64) -> BuiltinResult<()> {
        if !tolerance.is_finite() || tolerance < 0. {
            return Err(invalid("invalid borrowed genetic feasibility tolerance"));
        }
        self.maximum = maximum;
        self.tolerance = tolerance;
        Ok(())
    }
    async fn classify(
        &mut self,
        point: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Membership> {
        let (lower, upper) = self.bounds.original();
        if point.len() != lower.len() || !tolerance.is_finite() || tolerance < 0. {
            return Err(invalid("borrowed genetic point/tolerance extent changed"));
        }
        let mut in_box = true;
        for ((&x, &l), &u) in point.iter().zip(lower).zip(upper) {
            in_box &= x.is_finite() && x >= l && x <= u;
            cooperation.completed(1).await?;
        }
        let violation = if point.iter().all(|x| x.is_finite()) {
            self.linear
                .violation(point, cooperation)
                .await?
                .unwrap_or(f64::INFINITY)
        } else {
            f64::INFINITY
        };
        Ok(Membership {
            feasible: in_box && violation <= self.effective_tolerance(tolerance),
            violation,
        })
    }
    async fn repair(
        &mut self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Repair> {
        cooperation.completed(0).await?;
        if self.expired() {
            return Ok(Repair::TimeLimit);
        }
        if self
            .classify(point, self.tolerance, cooperation)
            .await?
            .feasible
        {
            return Ok(Repair::Unchanged);
        }
        if point.iter().any(|x| !x.is_finite()) {
            return Ok(Repair::Unverified);
        }
        let (lower, upper) = self.bounds.original();
        let mut initial = buffer(point.len())?;
        for ((&x, &l), &u) in point.iter().zip(lower).zip(upper) {
            initial.push(x.clamp(l, u));
            cooperation.completed(1).await?;
        }
        let candidate = match phase_one::feasible_start(
            initial,
            self.linear,
            self.effective_tolerance(self.tolerance),
            IDENTITY,
            || self.expired(),
            cooperation,
        )
        .await?
        {
            Ok(point) => point,
            Err(failure) => {
                return Ok(match failure.reason {
                    phase_one::Failure::Infeasible => Repair::Infeasible,
                    phase_one::Failure::Numerical => Repair::Unverified,
                    phase_one::Failure::TimeLimit => Repair::TimeLimit,
                })
            }
        };
        if self.expired() {
            return Ok(Repair::TimeLimit);
        }
        if !self
            .classify(&candidate, self.tolerance, cooperation)
            .await?
            .feasible
        {
            return Ok(Repair::Unverified);
        }
        let mut unchanged = true;
        for (a, b) in point.chunks(1024).zip(candidate.chunks(1024)) {
            unchanged &= a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            cooperation.completed(a.len() as u64).await?;
        }
        Ok(if unchanged {
            Repair::Unchanged
        } else {
            Repair::Point(candidate)
        })
    }
    fn row_scratch(&self, _: &ScalarCooperation) -> BuiltinResult<Vec<(usize, f64)>> {
        buffer(self.linear.row_capacity())
    }
    async fn original_row(
        &self,
        row: usize,
        scratch: &mut Vec<(usize, f64)>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.linear.original_row(row, scratch, cooperation).await
    }
}
