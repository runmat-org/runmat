//! Public residual sign and NNLS dual vector, not a bound-multiplier struct.
use super::*;
pub(super) fn finalize(outcome: model::Outcome, iterations: usize) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    let count = requested.unwrap_or(1);
    if count == 0 {
        return Ok(Value::output_list(Vec::new()));
    }
    let x = column(outcome.x)?;
    if requested.is_none() {
        return Ok(x);
    }
    let mut values = vec![x];
    if count >= 2 {
        values.push(Value::Num(outcome.resnorm));
    }
    if count >= 3 {
        values.push(column(outcome.residual)?);
    }
    if count >= 4 {
        values.push(Value::Num(outcome.flag as f64));
    }
    if count >= 5 {
        let mut information = runmat_value::UnboundedStructValueBuilder::new();
        information.insert("iterations", Value::Num(iterations as f64));
        information.insert("algorithm", Value::from(ALGORITHM));
        information.insert("message", Value::from(outcome.message));
        values.push(Value::Struct(information.finish()));
    }
    if count >= 6 {
        values.push(column(outcome.lambda)?);
    }
    Ok(crate::output_count::output_list_with_padding(count, values))
}
fn column(values: Vec<f64>) -> BuiltinResult<Value> {
    let rows = values.len();
    Tensor::new(values, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(invalid)
}
