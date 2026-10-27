//! Move public outputs once; failed certificates never manufacture duals.
use super::*;
pub(super) fn column(values: Vec<f64>) -> Value {
    let n = values.len();
    Tensor::new(values, vec![n, 1])
        .map(Value::Tensor)
        .unwrap_or_else(|_| empty_value())
}
pub(super) fn finalize(outcome: ConeOutcome) -> Value {
    let requested = crate::output_count::current_output_count();
    let count = requested.unwrap_or(1);
    if count == 0 {
        return Value::output_list(Vec::new());
    }
    let output = if count >= 4 {
        Some(Value::Struct(information(&outcome)))
    } else {
        None
    };
    let lambda = if count >= 5 {
        Some(outcome.lambda.map(multipliers).unwrap_or_else(empty_value))
    } else {
        None
    };
    let x = outcome.x.map(column).unwrap_or_else(empty_value);
    if requested.is_none() {
        return x;
    }
    let fval = outcome.fval.map(Value::Num).unwrap_or_else(empty_value);
    let flag = Value::Num(outcome.exitflag as f64);
    let mut values = vec![x];
    if count >= 2 {
        values.push(fval);
    }
    if count >= 3 {
        values.push(flag);
    }
    if let Some(output) = output {
        values.push(output);
    }
    if let Some(lambda) = lambda {
        values.push(lambda);
    }
    crate::output_count::output_list_with_padding(count, values)
}
fn information(outcome: &ConeOutcome) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields.insert("constrviolation", Value::Num(outcome.constrviolation));
    let unavailable = matches!(outcome.exitflag, -2 | -3 | -10);
    fields.insert(
        "algorithm",
        if unavailable {
            empty_value()
        } else {
            Value::from(ALGORITHM)
        },
    );
    fields.insert(
        "linearsolver",
        if unavailable {
            empty_value()
        } else {
            Value::from("qdldl")
        },
    );
    for (name, value) in [
        ("primalfeasibility", outcome.primalfeasibility),
        ("dualfeasibility", outcome.dualfeasibility),
        ("dualitygap", outcome.dualitygap),
    ] {
        fields.insert(
            name,
            if unavailable {
                empty_value()
            } else {
                Value::Num(value)
            },
        );
    }
    fields.finish()
}
fn multipliers(lambda: model::Lambda) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for (name, values) in [
        ("ineqlin", lambda.ineqlin),
        ("eqlin", lambda.eqlin),
        ("lower", lambda.lower),
        ("upper", lambda.upper),
        ("soc", lambda.soc),
    ] {
        fields.insert(name, column(values));
    }
    Value::Struct(fields.finish())
}
