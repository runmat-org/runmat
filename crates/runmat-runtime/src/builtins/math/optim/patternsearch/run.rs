//! Original problem ownership and public algorithm lifecycle. Numeric trials
//! and output observers have disjoint budgets and result-cardinality rules.
use super::{
    data, evaluation, input::Input, invalid, iteration, observation, options::Options, phase_one,
    random, state::Stop, timing::Deadline, IDENTITY,
};
use crate::builtins::math::optim::common::{
    self, constrained_numeric as numeric, ScalarCooperation,
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::OptimOptionField as Field;
mod controller;

pub(super) struct Outcome {
    pub engine: Option<iteration::Engine>,
    /// A genuine initial point returned before any objective evaluation. None
    /// is reserved for proven inconsistent bounds/linear infeasibility.
    pub initial: Option<Vec<f64>>,
    pub reason: Stop,
    pub objective_calls: usize,
    pub constraint_calls: usize,
    pub violation: Option<f64>,
}

mod preparation;
mod active;
type ResultTuple = (Outcome, Option<RootedValue>, Options);
#[expect(
    clippy::large_enum_variant,
    reason = "prepared solver state stays inline rather than adding an unbudgeted heap allocation"
)]
enum Preparation {
    Finished(ResultTuple),
    Ready(Ready),
}
struct Ready {
    engine: iteration::Engine,
    options: Options,
    linear: super::linear::Linear,
    domain: observation::Domain,
    policy: numeric::Policy,
    shape: Vec<usize>,
    scalar: bool,
    state: evaluation::State,
    deadline: Deadline,
    observers: observation::Observers,
    rng: Option<RootedValue>,
}
pub(super) fn solve<'a>(
    input: &'a Input<'_>,
    prepared: data::Prepared,
    options: Options,
    domain: observation::Domain,
    policy: numeric::Policy,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<ResultTuple>> + 'a {
    crate::execution::after_result(
        async move {
            let prepared =
                preparation::prepare(input, prepared, options, domain, policy, cooperation).await?;
            Ok((prepared, cooperation))
        },
        move |(prepared, cooperation)| match prepared {
            Preparation::Finished(result) => {
                futures::future::Either::Left(std::future::ready(Ok(result)))
            }
            Preparation::Ready(ready) => {
                futures::future::Either::Right(active::run(input, ready, cooperation))
            }
        },
    )
}
