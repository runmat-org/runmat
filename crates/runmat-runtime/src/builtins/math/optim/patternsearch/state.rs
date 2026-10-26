//! Accepted generations and measured changes. No original point is replayed
//! to construct observer metadata or a final result.
use super::{evaluation::Point, invalid};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    Mesh,
    Step,
    Function,
    Machine,
    Limit,
    Time,
    Observer,
    Numerical,
    Infeasible,
}
impl Stop {
    pub fn flag(self) -> i32 {
        match self {
            Self::Mesh => 1,
            Self::Step => 2,
            Self::Function => 3,
            Self::Machine => 4,
            Self::Limit | Self::Time => 0,
            Self::Observer => -1,
            Self::Infeasible => -2,
            Self::Numerical => -4,
        }
    }
    pub fn message(self) -> &'static str {
        match self {
            Self::Mesh => "The completed mesh poll and original constraint checks meet the mesh tolerance; with nonlinear constraints the augmented-Lagrangian complementarity check also meets its tolerance.",
            Self::Step => "An unsuccessful poll has sufficiently small mesh size and measured change in x, with the original constraints feasible.",
            Self::Function => "An unsuccessful poll has sufficiently small mesh size and measured change in objective value, with the original constraints feasible.",
            Self::Machine => "No represented feasible poll step changes the current point, and the original constraints are feasible.",
            Self::Limit => "The objective-evaluation or completed-iteration limit was reached.",
            Self::Time => "The original invocation clock reached MaxTime after a completed poll.",
            Self::Observer => "An output or plot function requested termination.",
            Self::Infeasible => "The original linear constraints or bounds are infeasible; no objective callback was evaluated.",
            Self::Numerical => "A reliable feasible generating set, represented step, or augmented-Lagrangian transition could not be obtained; this is not an infeasibility certificate.",
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Change {
    pub x: f64,
    pub f: f64,
}
/// Only original x and f are retained across a nonlinear inner solve. Old
/// constraint rows are dropped, not copied into a second point generation.
pub(super) struct Anchor {
    x: Vec<f64>,
    f: f64,
}
impl Anchor {
    pub fn take(point: Point) -> Self {
        Self {
            x: point.x,
            f: point.f,
        }
    }
    pub async fn change(
        &self,
        point: &Point,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Change> {
        measure(&self.x, self.f, point, cooperation).await
    }
}
pub(super) async fn measure(
    x: &[f64],
    f: f64,
    point: &Point,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Change> {
    if x.len() != point.x.len() {
        return Err(invalid("accepted point extent changed"));
    }
    let mut distance = 0.0f64;
    for (&old, &new) in x.iter().zip(&point.x) {
        distance = distance.hypot(new - old);
        cooperation.completed(1).await?;
    }
    // Overflow is retained as an infinite measured change, never a small-step
    // certificate. Both original objective values themselves are finite.
    Ok(Change {
        x: distance,
        f: (point.f - f).abs(),
    })
}

pub(super) async fn nonlinear_violation(
    point: &Point,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    let mut maximum = 0.0f64;
    for &value in &point.inequality {
        maximum = maximum.max(value);
        cooperation.completed(1).await?;
    }
    for &value in &point.equality {
        maximum = maximum.max(value.abs());
        cooperation.completed(1).await?;
    }
    Ok(maximum)
}
