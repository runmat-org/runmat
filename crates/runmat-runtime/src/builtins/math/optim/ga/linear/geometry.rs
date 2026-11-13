//! Borrowed original-domain capability used by the one genetic step kernel.
//! Implementations retain the domain authority; numerical proposals do not
//! authorize publication or become a second model.
use super::super::super::intlinprog::feasibility::{Membership, Repair};
use super::super::{common::global_time::TimeLimit, ScalarCooperation};
use crate::BuiltinResult;
use num_rational::BigRational;

pub(in crate::builtins::math::optim) trait Geometry {
    fn bounds(&self) -> (&[f64], &[f64]);
    fn integers(&self) -> &[usize];
    fn rhs(&self) -> &[f64];
    fn equalities(&self) -> usize;
    fn repair_nodes(&self) -> usize;
    /// An enclosing domain may impose a tighter deadline on the SAME clock.
    fn time_limit_reached(
        &self,
        timing: &super::super::common::solver_timing::SolverTiming,
        maximum: TimeLimit,
    ) -> bool {
        maximum.reached(timing.elapsed())
    }
    fn retune(&mut self, maximum: TimeLimit, tolerance: f64) -> BuiltinResult<()>;
    async fn repair(
        &mut self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Repair>;
    async fn classify(
        &mut self,
        point: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Membership>;
    fn row_scratch(&self, cooperation: &ScalarCooperation) -> BuiltinResult<Vec<(usize, f64)>>;
    async fn original_row(
        &self,
        row: usize,
        scratch: &mut Vec<(usize, f64)>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()>;
    async fn step(
        &mut self,
        parent: &[f64],
        direction: &[BigRational],
        controls: super::step::Controls<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<super::step::Step> {
        super::step::candidate(self, parent, direction, controls, cooperation).await
    }
}
