//! Find a linearly feasible starting point without calling the objective or
//! replacing the direct-search algorithm with a public optimizer callback.
use super::{buffer, linear::Linear};
use crate::builtins::math::optim::common::{conic, ScalarCooperation};
use crate::{context::RuntimeContext, BuiltinResult};
use clarabel::solver::{DefaultInfo, DefaultSettings};
use runmat_builtins::BuiltinCatalogIdentity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Failure {
    Infeasible,
    Numerical,
    TimeLimit,
}
pub(super) struct Unstarted {
    pub initial: Vec<f64>,
    pub reason: Failure,
}

struct Control<'a, D> {
    accuracy: f64,
    used: u32,
    maximum: u32,
    deadline: &'a D,
}
impl<D: Fn() -> bool> conic::solve::IterationControl for Control<'_, D> {
    fn accuracy(&self) -> f64 {
        self.accuracy
    }
    fn remaining(&self) -> u32 {
        self.maximum.saturating_sub(self.used)
    }
    fn progress_control(&mut self) -> clarabel::solver::ProgressControl {
        if (self.deadline)() {
            clarabel::solver::ProgressControl::Stop(clarabel::solver::ProgressStop::TimeLimit)
        } else {
            clarabel::solver::ProgressControl::Continue
        }
    }
    fn record(
        &mut self,
        previous: &mut u32,
        info: &DefaultInfo<f64>,
        _scale: f64,
        _runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()> {
        self.used = self
            .used
            .saturating_add(info.iterations.saturating_sub(*previous));
        *previous = info.iterations;
        Ok(())
    }
}

pub(super) async fn feasible_start<D: Fn() -> bool>(
    initial: Vec<f64>,
    linear: &Linear,
    tolerance: f64,
    identity: BuiltinCatalogIdentity,
    deadline: D,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Result<Vec<f64>, Unstarted>> {
    if deadline() {
        return Ok(Err(Unstarted {
            initial,
            reason: Failure::TimeLimit,
        }));
    }
    if linear.feasible(&initial, tolerance, cooperation).await? {
        return Ok(Ok(initial));
    }
    let mut zero = buffer(initial.len())?;
    for _ in &initial {
        zero.push(0.0);
        cooperation.completed(1).await?;
    }
    let model = conic::Model {
        p: None,
        a: &linear.matrix,
        b: &linear.rhs,
        f: &zero,
        cones: &linear.cones,
        identity,
        objective_scale: 1.0,
    };
    // This is the backend's ordinary feasibility guard, not an invented
    // dimension cliff or an objective-evaluation allowance. Exhaustion cannot
    // establish original infeasibility; only the exact certificate can.
    let mut control = Control {
        accuracy: tolerance,
        used: 0,
        maximum: DefaultSettings::<f64>::default().max_iter,
        deadline: &deadline,
    };
    let failure = match conic::solve::candidate(&model, &initial, &mut control, cooperation).await?
    {
        conic::solve::Candidate::Stopped {
            kind: conic::solve::StopKind::TimeLimit,
            ..
        } => Failure::TimeLimit,
        conic::solve::Candidate::Converged { x, .. }
        | conic::solve::Candidate::Stopped { x: Some(x), .. } => {
            if linear.feasible(&x, tolerance, cooperation).await? {
                return Ok(Ok(x));
            } else {
                Failure::Numerical
            }
        }
        conic::solve::Candidate::CertifiedInfeasible => Failure::Infeasible,
        _ => Failure::Numerical,
    };
    Ok(Err(Unstarted {
        initial,
        reason: failure,
    }))
}
