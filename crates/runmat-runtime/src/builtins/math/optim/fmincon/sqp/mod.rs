//! Shared sequential-quadratic major iterations. The typed strategy selects
//! distinct conic or primal working-set quadratic subproblem mechanisms.
use super::hessian as bfgs;
mod initialization;
mod iteration;
mod merit;
mod restoration;
mod transition;
pub(super) use super::outcome::{Outcome, Parameters, Termination};
use super::quadratic as subproblem;
pub(super) use iteration::minimize;
pub(super) use iteration::minimize_staged;
pub(super) use iteration::minimize_view;
pub(super) use iteration::minimize_with;
#[derive(Clone, Copy)]
pub(super) enum Strategy {
    Conic,
    WorkingSet,
}
