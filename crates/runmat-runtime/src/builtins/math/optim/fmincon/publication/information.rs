//! Same selected solver information for raw and modeled result publication.
use super::{Outcome, ReflectiveInfo, SolverTiming, Value};
use crate::BuiltinResult;
pub(in crate::builtins::math::optim::fmincon) fn value(
    outcome: &Outcome,
    timing: Option<&SolverTiming>,
    reflective: Option<ReflectiveInfo>,
) -> BuiltinResult<Value> {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in [
        ("iterations", outcome.iterations),
        ("funcCount", outcome.counts.points),
        ("objectiveCallCount", outcome.counts.objective_calls),
        ("constraintCallCount", outcome.counts.constraint_calls),
    ] {
        fields.insert(name, Value::Num(value as f64));
    }
    for (name, value) in [
        ("constrviolation", outcome.violation),
        ("firstorderopt", outcome.optimality),
    ] {
        fields.insert(
            name,
            value.map(Value::Num).unwrap_or_else(super::super::empty),
        );
    }
    fields.insert("stepsize", Value::Num(outcome.step));
    fields.insert("algorithm", Value::from(outcome.algorithm.name()));
    fields.insert("message", Value::from(outcome.termination.description()));
    if let Some(info) = reflective {
        fields.insert("cgiterations", Value::Num(info.cg_iterations as f64));
        fields.insert("trustregionradius", Value::Num(info.radius));
    }
    fields.insert(
        "wallclocktime",
        Value::Num(
            timing
                .ok_or_else(SolverTiming::unavailable)?
                .elapsed()
                .as_secs_f64(),
        ),
    );
    Ok(Value::Struct(fields.finish()))
}
