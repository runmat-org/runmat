//! Five actual final outputs, rooted before predecessor generations retire.
use super::{inputs, iteration::Outcome, population::capacity, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::Value;
pub(super) fn finalize(
    outcome: Outcome,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let requested = crate::output_count::current_output_count_in(cooperation.runtime());
    let count = requested.unwrap_or(1);
    let mut outputs = capacity(count, cooperation)?;
    if count > 0 {
        outputs.push(
            outcome
                .point
                .as_ref()
                .map(|v| v.value().clone())
                .unwrap_or_else(inputs::empty),
        );
    }
    if count >= 2 {
        outputs.push(
            outcome
                .fitness
                .map(Value::Num)
                .unwrap_or_else(inputs::empty),
        );
    }
    if count >= 3 {
        outputs.push(Value::Num(outcome.reason.flag() as f64));
    }
    if count >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "iterations",
            Value::Num(outcome.state.as_ref().map_or(0, |v| v.iteration) as f64),
        );
        fields.insert("funccount", Value::Num(outcome.counts.points as f64));
        fields.insert(
            "swarmObjectiveCallCount",
            Value::Num(outcome.counts.callbacks as f64),
        );
        fields.insert("message", Value::from(outcome.reason.message()));
        fields.insert(
            "hybridflag",
            outcome
                .hybrid_flag
                .map(Value::Num)
                .unwrap_or_else(inputs::empty),
        );
        fields.insert(
            "rngstate",
            outcome
                .rng
                .as_ref()
                .map(|v| v.value().clone())
                .unwrap_or_else(inputs::empty),
        );
        outputs.push(Value::Struct(fields.finish()));
    }
    if count >= 5 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert(
            "X",
            outcome
                .state
                .as_ref()
                .map(|v| v.population.value())
                .unwrap_or_else(inputs::empty),
        );
        fields.insert(
            "Fval",
            outcome
                .state
                .as_ref()
                .map(|v| v.fitness.value())
                .unwrap_or_else(inputs::empty),
        );
        outputs.push(Value::Struct(fields.finish()));
    }
    let sequence = if requested.is_none() {
        runmat_value::ValueSequence::single(outputs.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(outputs)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match cooperation.runtime() {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => crate::execution::RootedValueSequence::new(sequence)
            .map_err(|source| source.into_runtime_error()),
    }
}
