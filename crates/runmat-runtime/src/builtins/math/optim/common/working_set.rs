//! One feasible primal blocking/Bland/release controller, with domain-owned targets.
use super::{solver_workspace as workspace, ScalarCooperation};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
mod iteration;
mod recession;
pub(crate) mod rows;
pub(crate) mod warm;
pub(crate) use iteration::minimize;
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub a: &'a CscMatrix<f64>,
    pub b: &'a [f64],
    pub equalities: usize,
}
pub(crate) struct Step {
    pub direction: Vec<f64>,
    pub multipliers: Vec<f64>,
}
/// A checked equality-subproblem proposal. A recession direction has no
/// artificial unit step: the next original inactive row determines its extent.
pub(crate) enum EqualityTarget {
    Point(Step),
    Recession(Vec<f64>),
}
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub feasibility: f64,
    pub dual: f64,
    pub pivots: usize,
}
#[derive(Clone, Copy)]
pub(crate) struct Change {
    pub length: f64,
    pub reference: f64,
}
pub(crate) enum Stop {
    IterationLimit,
    TimeLimit,
    ObjectiveLimit,
    StepStagnation,
    Unverified,
}
pub(crate) enum Outcome {
    Optimal(Step),
    Stopped {
        point: Vec<f64>,
        reason: Stop,
    },
    Numerical {
        point: Vec<f64>,
    },
    /// Candidate only. The numerical controller does not issue a source proof.
    Unbounded {
        point: Vec<f64>,
        direction: Vec<f64>,
    },
}
pub(crate) trait Target {
    /// Optional real warm state, checked after projection/feasibility on x.
    /// The returned vector becomes the controller's existing working table.
    async fn initial_active(
        &mut self,
        _rows: Rows<'_>,
        _x: &[f64],
        _activity: &[f64],
        _c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<bool>>> {
        Ok(None)
    }
    /// Move the completed working table, never reconstruct it from duals.
    fn retain_active(&mut self, _active: Vec<bool>) {}
    fn begin(&mut self) -> BuiltinResult<()> {
        Ok(())
    }
    async fn equality(
        &mut self,
        a: &CscMatrix<f64>,
        b: &[f64],
        x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<EqualityTarget>>;
    async fn checked(
        &mut self,
        rows: Rows<'_>,
        x: &[f64],
        z: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn advanced(
        &mut self,
        _x: &[f64],
        _c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Stop>> {
        Ok(None)
    }
    fn small_step(&self, _change: Change) -> bool {
        false
    }
}
