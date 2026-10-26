//! Four documented outputs, rooted before original generations retire.
use super::{
    inputs,
    iteration::{Generation, Outcome},
    ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::Value;
pub(super) fn finalize(
    mut outcome: Outcome,
    c: &ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let requested = crate::output_count::current_output_count_in(c.runtime());
    let count = requested.unwrap_or(1);
    let mut values = Vec::with_capacity(count);
    if count > 0 {
        values.push(match &outcome.generation {
            Generation::Measured(state) => state.best.value().clone(),
            Generation::Unmeasured { point, .. } => point.value().clone(),
            Generation::Infeasible => inputs::empty(),
        });
    }
    if count >= 2 {
        values.push(match &outcome.generation {
            Generation::Measured(state) => Value::Num(state.best_f),
            _ => inputs::empty(),
        });
    }
    if count >= 3 {
        values.push(Value::Num(outcome.reason.flag()));
    }
    if count >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        let iteration = match &outcome.generation {
            Generation::Measured(state) => state.iteration,
            _ => 0,
        };
        fields.insert("iterations", Value::Num(iteration as f64));
        fields.insert("funccount", Value::Num(outcome.count as f64));
        fields.insert("message", Value::from(outcome.reason.message()));
        fields.insert("rngstate", outcome.rng.value().clone());
        fields.insert(
            "problemtype",
            Value::from(if outcome.bounded {
                "boundconstraints"
            } else {
                "unconstrained"
            }),
        );
        let temperature = match &mut outcome.generation {
            Generation::Measured(state) => Some(std::mem::take(&mut state.temperature)),
            Generation::Unmeasured { temperature, .. } => Some(std::mem::take(temperature)),
            Generation::Infeasible => None,
        };
        let temperature = temperature
            .map(|values| {
                let n = values.len();
                runmat_value::Tensor::new(values, vec![n, 1])
                    .map(Value::Tensor)
                    .map_err(super::invalid)
            })
            .transpose()?
            .unwrap_or_else(inputs::empty);
        fields.insert("temperature", temperature);
        fields.insert("totaltime", Value::Num(outcome.elapsed));
        values.push(Value::Struct(fields.finish()));
    }
    let sequence = if requested.is_none() {
        runmat_value::ValueSequence::single(values.remove(0))
    } else {
        runmat_value::ValueSequence::comma_separated(values)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    match c.runtime() {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => {
            crate::execution::RootedValueSequence::new(sequence).map_err(|e| e.into_runtime_error())
        }
    }
}
