//! Original accepted numeric backing moves into the public x. Callback and
//! RNG handles remain rooted until the successor result forest is published.
use super::{buffer, controls::Controls, run::Outcome};
use crate::builtins::math::optim::common::{self, output_callback, ScalarCooperation};
use crate::{
    execution::{RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::{Value, ValueSequence};

pub(super) async fn finalize(
    mut outcome: Outcome,
    rng: Option<RootedValue>,
    objective: &Value,
    shape: &[usize],
    scalar: bool,
    constrained: bool,
    problem_type: &str,
    controls: &Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    cooperation.completed(0).await?;
    let requested = crate::output_count::current_output_count_in(cooperation.runtime());
    let count = requested.unwrap_or(1);
    let iterations = outcome
        .engine
        .as_ref()
        .map_or(0, |engine| engine.iterations);
    let mesh = outcome
        .engine
        .as_ref()
        .map_or(controls.mesh.initial, |engine| engine.mesh.size());
    let point = outcome.engine.take().map(|engine| engine.into_point());
    let f = point.as_ref().map(|point| point.f);
    let x = point.map(|point| point.x).or(outcome.initial);
    let mut outputs = buffer(count)?;
    if count > 0 {
        outputs.push(match x {
            Some(x) => common::vector_to_value("patternsearch", x, shape, scalar)?,
            None => output_callback::empty(cooperation.runtime())?,
        });
    }
    if count >= 2 {
        outputs.push(match f {
            Some(f) => Value::Num(f),
            None => output_callback::empty(cooperation.runtime())?,
        });
    }
    if count >= 3 {
        outputs.push(Value::Num(outcome.reason.flag() as f64));
    }
    if count >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("function", objective.clone());
        fields.insert("problemtype", Value::from(problem_type));
        fields.insert("pollmethod", Value::from(controls.polling.name()));
        fields.insert(
            "searchmethod",
            match controls.search.filter(|search| {
                !controls
                    .polling
                    .method()
                    .is_some_and(|method| search.redundant(method))
            }) {
                Some(search) => Value::from(search.name()),
                None => output_callback::empty(cooperation.runtime())?,
            },
        );
        fields.insert("iterations", Value::Num(iterations as f64));
        fields.insert("funccount", Value::Num(outcome.objective_calls as f64));
        fields.insert(
            "constraintcallcount",
            Value::Num(outcome.constraint_calls as f64),
        );
        fields.insert("meshsize", Value::Num(mesh));
        fields.insert(
            "maxconstraint",
            match outcome.violation.filter(|_| constrained) {
                Some(violation) => Value::Num(violation),
                None => output_callback::empty(cooperation.runtime())?,
            },
        );
        fields.insert(
            "rngstate",
            match &rng {
                Some(rng) => rng.value().clone(),
                None => output_callback::empty(cooperation.runtime())?,
            },
        );
        fields.insert("message", Value::from(outcome.reason.message()));
        outputs.push(Value::Struct(fields.finish()));
    }
    let sequence = if requested.is_none() {
        ValueSequence::single(outputs.remove(0))
    } else {
        ValueSequence::comma_separated(outputs)
    }
    .map_err(crate::sequence::sequence_error_to_runtime)?;
    let result = match cooperation.runtime() {
        Some(runtime) => RootedValueSequence::in_context(sequence, runtime),
        None => RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error()),
    };
    drop(rng);
    result
}
