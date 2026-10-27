//! Actual original-row duals and truthful empty/numerical failure outputs.
use super::*;
pub(super) fn finalize(outcome: Outcome, timing: Option<SolverTiming>) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    finalize_selected(outcome, timing, requested, |point| {
        point
            .map(column)
            .transpose()
            .map(|value| value.unwrap_or_else(empty))
    })
}
pub(super) fn finalize_selected(
    outcome: Outcome,
    timing: Option<SolverTiming>,
    requested: Option<usize>,
    first: impl FnOnce(Option<Vec<f64>>) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let count = requested.unwrap_or(1);
    if count == 0 {
        return Ok(Value::output_list(Vec::new()));
    }
    let information = if count >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        for (name, value) in [
            ("iterations", outcome.iterations as f64),
            ("constrviolation", outcome.violation),
            ("firstorderopt", outcome.optimality),
            ("dualitygap", outcome.gap),
        ] {
            fields.insert(name, Value::Num(value));
        }
        fields.insert("algorithm", Value::from(compiled::algorithm(&outcome)));
        fields.insert(
            "wallclocktime",
            Value::Num(
                timing
                    .as_ref()
                    .ok_or_else(SolverTiming::unavailable)?
                    .elapsed()
                    .as_secs_f64(),
            ),
        );
        fields.insert("linearsolver", Value::from("sparse"));
        fields.insert("message", Value::from(outcome.message));
        Some(Value::Struct(fields.finish()))
    } else {
        None
    };
    let x = first(outcome.x)?;
    if requested.is_none() {
        return Ok(x);
    }
    let mut values = vec![x];
    if count >= 2 {
        values.push(outcome.fval.map(Value::Num).unwrap_or_else(empty));
    }
    if count >= 3 {
        values.push(Value::Num(outcome.flag as f64));
    }
    if let Some(information) = information {
        values.push(information);
    }
    if count >= 5 {
        values.push(match outcome.lambda {
            None => empty(),
            Some(lambda) => {
                let mut fields = runmat_value::UnboundedStructValueBuilder::new();
                for (name, values) in [
                    ("ineqlin", lambda.ineqlin),
                    ("eqlin", lambda.eqlin),
                    ("lower", lambda.lower),
                    ("upper", lambda.upper),
                ] {
                    fields.insert(name, column(values)?);
                }
                Value::Struct(fields.finish())
            }
        });
    }
    Ok(crate::output_count::output_list_with_padding(count, values))
}
fn column(values: Vec<f64>) -> BuiltinResult<Value> {
    let n = values.len();
    Tensor::new(values, vec![n, 1])
        .map(Value::Tensor)
        .map_err(invalid)
}
