//! Value checks for the shared typed MILP option profile; no solver dispatch.
use super::*;
use crate::builtins::math::optim::common;
use state::PreparedOptionValue;
pub(super) fn prepared(
    field: OptimOptionField,
    value: &Value,
) -> BuiltinResult<PreparedOptionValue<'_>> {
    let error = || {
        optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported or out-of-range intlinprog option {}",
                field.name()
            ),
        )
    };
    if matches!(value,Value::Tensor(v) if v.is_empty()) {
        return Ok(PreparedOptionValue::Callback(value));
    }
    if matches!(
        field,
        OptimOptionField::OutputFcn | OptimOptionField::PlotFcn
    ) {
        common::output_callback::validate_for(value, common::output_callback::milp_form(field))?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if field == OptimOptionField::Display {
        return Ok(PreparedOptionValue::Display(display_value(
            Solver::Intlinprog,
            value,
        )?));
    }
    if !matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
        return Err(error());
    }
    if matches!(
        field,
        OptimOptionField::MaxNodes | OptimOptionField::MaxFeasiblePoints
    ) {
        let positive_infinity = matches!(value,Value::Num(v) if *v==f64::INFINITY)
            || matches!(value,Value::Tensor(v) if v.len()==1 && v.integer_storage().is_none() && tensor::tensor_value_f64(v,0)==f64::INFINITY);
        if positive_infinity && field == OptimOptionField::MaxFeasiblePoints {
            return Ok(PreparedOptionValue::Number(f64::INFINITY));
        }
        let count = positive_integer_scalar(field, value, Solver::Intlinprog)?;
        return Ok(tensor::scalar_integer_value(value)
            .map(PreparedOptionValue::Integer)
            .unwrap_or(PreparedOptionValue::Number(count as f64)));
    }
    let parsed = match value {
        Value::Num(v) => *v,
        Value::Int(v) => v.to_exact_f64().ok_or_else(error)?,
        Value::Tensor(v)
            if v.len() == 1
                && !crate::builtins::common::validation::value_has_native_integer_class(value) =>
        {
            tensor::tensor_value_f64(v, 0)
        }
        Value::Tensor(v) if v.len() == 1 => tensor::scalar_integer_value(value)
            .and_then(|v| v.to_exact_f64())
            .ok_or_else(error)?,
        _ => return Err(error()),
    };
    if !field.intlinprog_scalar_range(parsed) {
        return Err(error());
    }
    Ok(PreparedOptionValue::Number(parsed))
}
