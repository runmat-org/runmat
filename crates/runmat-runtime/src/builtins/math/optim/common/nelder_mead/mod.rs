//! One Nelder-Mead mathematical owner shared by public and auxiliary search.
use super::{output_callback::OutputState, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
mod model;
mod geometry;
mod iteration;
pub(in crate::builtins::math::optim) use geometry::geometry;
pub(in crate::builtins::math::optim) use iteration::minimize;
pub(in crate::builtins::math::optim) use model::{
    Controls, Events, Initial, Measurement, Oracle, OracleStop, Outcome, Point, Policy, Procedure,
    Reason,
};

#[cfg(test)]
mod tests;
