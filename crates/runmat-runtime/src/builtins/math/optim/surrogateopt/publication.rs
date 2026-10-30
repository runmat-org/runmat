//! Final outputs are rooted before the controller and its source generations
//! retire. A missing measurement is never replaced by a modeled value.
use super::{
    controller::{Controller, Stop},
    public_values as values,
};
use crate::{
    builtins::math::optim::common::ScalarCooperation,
    execution::{RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::Value;
mod inconsistent;
pub(super) use inconsistent::publish as inconsistent;

pub(super) fn flag(controller: &Controller<'_>, stop: Stop) -> i32 {
    if stop == Stop::Observer {
        return -1;
    }
    if stop == Stop::Observer {
        return -1;
    }
    let feasible = controller
        .campaign
        .tracker()
        .best()
        .and_then(|i| {
            controller
                .campaign
                .history()
                .trial(i)
                .observation
                .measurement()
        })
        .is_some_and(|v| v.nonlinear.violated == 0);
    if !feasible {
        return -2;
    }
    match stop {
        Stop::ObjectiveLimit => 1,
        Stop::Unique => 10,
        Stop::TooFewSites => 3,
        Stop::FunctionLimit | Stop::TimeLimit => 0,
        Stop::LinearInfeasible => -2,
        Stop::Observer => -1,
    }
}
pub(super) fn message(stop: Stop, flag: i32) -> &'static str {
    if flag == -2 {
        return "No feasible point was found.";
    }
    match stop {
        Stop::FunctionLimit => "The maximum number of function evaluations was reached.",
        Stop::TimeLimit => "The maximum elapsed time was reached.",
        Stop::ObjectiveLimit => "A feasible objective value is below ObjectiveLimit.",
        Stop::Observer => "An output or plot function stopped the optimization.",
        Stop::TooFewSites => "Too few new feasible points were available to continue.",
        Stop::Unique => "The bounds and linear equalities have a unique feasible point.",
        Stop::LinearInfeasible => "The bounds, integer and linear constraints are infeasible.",
    }
}
pub(super) async fn finalize(
    controller: &Controller<'_>,
    stop: Stop,
    initial_rng: &RootedValue,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    controller.clock.checked_elapsed()?;
    let requested = crate::output_count::current_output_count_in(c.runtime());
    let count = requested.unwrap_or(1);
    if count > 5 {
        return Err(super::invalid("at most five outputs are available"));
    }
    let history = controller.campaign.history();
    let trial = controller
        .campaign
        .tracker()
        .best()
        .map(|i| history.trial(i));
    let status = flag(controller, stop);
    let mut outputs = values::capacity(count)?;
    if count > 0 {
        outputs.push(values::point(trial, c).await?);
    }
    if count > 1 {
        outputs.push(values::objective(trial, c)?);
    }
    if count > 2 {
        outputs.push(Value::Num(status as f64));
    }
    if count > 3 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("funccount", Value::Num(controller.counts().1 as f64));
        fields.insert(
            "elapsedtime",
            Value::Num(controller.clock.elapsed().as_secs_f64()),
        );
        fields.insert(
            "message",
            Value::string(crate::call::argument_text::copy_str(
                message(stop, status),
                c.runtime(),
            )?),
        );
        fields.insert("constrviolation", values::violation(trial, c)?);
        fields.insert("ineq", values::inequalities(trial, c).await?);
        fields.insert("rngstate", initial_rng.value().clone());
        outputs.push(Value::Struct(fields.finish()));
    }
    if count > 4 {
        outputs.push(values::trials(history, c).await?);
    }
    root(outputs, requested, c)
}
fn root(
    mut outputs: Vec<Value>,
    requested: Option<usize>,
    c: &ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let sequence = if requested.is_none() {
        runmat_value::ValueSequence::single(outputs.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(outputs)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match c.runtime() {
        Some(runtime) => RootedValueSequence::in_context(sequence, runtime),
        None => RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error()),
    }
}
