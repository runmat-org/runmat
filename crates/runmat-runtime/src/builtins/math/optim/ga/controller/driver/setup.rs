//! One-time RNG and observer setup does not reserve its constructor layout
//! in the active evolution/notification driver's poll frame.
use super::{Observing, ScalarCooperation, SolverTiming};
use crate::builtins::math::optim::ga::options::Options;
use crate::{execution::RootedValue, BuiltinResult};

pub(super) struct Initialized {
    pub rng: RootedValue,
    pub observers: Observing,
}
pub(super) async fn initialize(
    state: Option<crate::builtins::common::random::RngSnapshot>,
    options: &Options,
    timing: &SolverTiming,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Initialized> {
    let rng = super::super::super::random::begin(state, cooperation).await?;
    let observers = Observing::new(options, timing, cooperation).await?;
    Ok(Initialized { rng, observers })
}
