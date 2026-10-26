//! Infeasible-start primal-dual logarithmic-barrier family, separate from SQP.
mod globalization;
mod initialization;
mod iteration;
mod newton;
mod state;
mod transition;
pub(super) use iteration::minimize;
