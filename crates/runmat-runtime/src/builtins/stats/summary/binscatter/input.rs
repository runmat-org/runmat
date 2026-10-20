//! Direct vectors and table-variable selectors for chart observations.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) fn admit_host_visible(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<ParsedInputs>> {
    for value in args.iter() {
        if let Value::GpuTensor(handle) = value {
            if handle.physical().storage() != GpuTensorStorage::Real {
                return Err(invalid("binscatter: x and y must be real numeric vectors"));
            }
            if handle.shape().iter().filter(|&&extent| extent > 1).count() > 1 {
                return Err(invalid("binscatter: x and y must be real numeric vectors"));
            }
        }
    }
    if args
        .iter()
        .all(|value| !matches!(value, Value::GpuTensor(_)))
    {
        return parse_inputs(args).map(Some);
    }
    if args.len() == 2 {
        for value in args.iter() {
            if !matches!(
                value,
                Value::GpuTensor(_)
                    | Value::Num(_)
                    | Value::Int(_)
                    | Value::Tensor(_)
                    | Value::Bool(_)
                    | Value::LogicalArray(_)
            ) {
                return Err(invalid("binscatter: x and y must be real numeric vectors"));
            }
            let shape = match value {
                Value::Tensor(tensor) => Some(tensor.shape()),
                Value::LogicalArray(array) => Some(array.shape.as_slice()),
                _ => None,
            };
            if shape.is_some_and(|dims| dims.iter().filter(|&&extent| extent > 1).count() > 1) {
                return Err(invalid("binscatter: x and y must be real numeric vectors"));
            }
        }
        let lengths: Option<Vec<usize>> = args
            .iter()
            .map(|value| match value {
                Value::GpuTensor(handle) => handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |acc, dim| acc.checked_mul(*dim)),
                Value::Tensor(tensor) => Some(tensor.len()),
                Value::LogicalArray(array) => Some(array.len()),
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(1),
                _ => None,
            })
            .collect();
        if let Some(lengths) = lengths {
            if lengths[0] != lengths[1] {
                return Err(invalid(
                    "binscatter: x and y must contain the same number of elements",
                ));
            }
        } else {
            return Err(invalid("binscatter: input shape overflows"));
        }
    } else if let Value::Object(table) = &args[0] {
        let names = crate::builtins::table::table_variable_names_from_object(table)
            .map_err(|err| invalid_with_source(err.message.clone(), err))?;
        for (selector, role) in [(&args[1], "xvar"), (&args[2], "yvar")] {
            if let Value::GpuTensor(handle) = selector {
                let count = handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                    .ok_or_else(|| invalid("binscatter: table selector shape overflows"))?;
                if count != 1 {
                    return Err(invalid(format!(
                        "binscatter: {role} must be a scalar table variable index"
                    )));
                }
            } else {
                let name = table_variable_name(selector, &names, role)?;
                if let Some(variables) = crate::builtins::table::table_variables_ref(table)
                    .map_err(|err| invalid_with_source(err.message.clone(), err))?
                {
                    if let Some(value) = variables.fields.get(&name) {
                        validate_selected_host_data(value, &name)?;
                    }
                }
            }
        }
    }
    Ok(None)
}

fn validate_selected_host_data(value: &Value, name: &str) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => None,
        _ => return Err(invalid(format!("binscatter: {name} must be numeric"))),
    };
    if shape.is_some_and(|dims| dims.iter().filter(|&&extent| extent > 1).count() > 1) {
        return Err(invalid(format!(
            "binscatter: {name} must be a real numeric vector"
        )));
    }
    Ok(())
}

pub(super) fn parse_inputs(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedInputs> {
    if args.len() < 2 {
        return Err(invalid("binscatter: expected x and y inputs"));
    }
    if let Value::Object(object) = &args[0] {
        if crate::builtins::table::is_tabular_object(object) {
            if args.len() < 3 {
                return Err(invalid("binscatter: expected table, xvar, and yvar"));
            }
            let x_data = table_variable_numeric_data(object, &args[1], "xvar")?;
            let y_data = table_variable_numeric_data(object, &args[2], "yvar")?;
            ensure_compatible_lengths(&x_data, &y_data)?;
            return Ok(ParsedInputs { x_data, y_data });
        }
    }

    let x_value = args[0].clone();
    let y_value = args[1].clone();
    let x_data = numeric_data_from_value(x_value, "x")?;
    let y_data = numeric_data_from_value(y_value, "y")?;
    ensure_compatible_lengths(&x_data, &y_data)?;
    Ok(ParsedInputs { x_data, y_data })
}

fn table_variable_numeric_data(
    object: &ObjectInstance,
    selector: &Value,
    context: &str,
) -> BuiltinResult<Tensor> {
    let names = crate::builtins::table::table_variable_names_from_object(object)
        .map_err(|err| invalid_with_source(err.message.clone(), err))?;
    let name = table_variable_name(selector, &names, context)?;
    let variables = crate::builtins::table::table_variables(object)
        .map_err(|err| invalid_with_source(err.message.clone(), err))?;
    let value = variables
        .fields
        .get(&name)
        .cloned()
        .ok_or_else(|| invalid(format!("binscatter: table variable '{name}' not found")))?;
    numeric_data_from_value(value, &name)
}

fn table_variable_name(selector: &Value, names: &[String], context: &str) -> BuiltinResult<String> {
    if let Some(name) = tensor::value_to_string(selector) {
        if names.iter().any(|candidate| candidate == &name) {
            return Ok(name);
        }
        return Err(invalid(format!(
            "binscatter: {context} table variable '{name}' not found"
        )));
    }
    if let Some(index) = numeric_selector_index(selector)? {
        if let Some(name) = names.get(index) {
            return Ok(name.clone());
        }
        return Err(invalid(format!(
            "binscatter: {context} table variable index exceeds table width"
        )));
    }
    Err(invalid(format!(
        "binscatter: {context} must be a table variable name or index"
    )))
}

pub(super) fn numeric_selector_index(selector: &Value) -> BuiltinResult<Option<usize>> {
    if let Some(value) = integer_scalar(selector) {
        let Some(index) = value.try_to_usize().and_then(|value| value.checked_sub(1)) else {
            return Err(invalid(
                "binscatter: table variable index must be a positive integer",
            ));
        };
        return Ok(Some(index));
    }
    let raw = match selector {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    let Some(value) = raw else {
        return Ok(None);
    };
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(invalid(
            "binscatter: table variable index must be a positive integer",
        ));
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return Err(invalid(
            "binscatter: table variable index exceeds platform limits",
        ));
    }
    Ok(Some(value as usize - 1))
}

pub(super) fn numeric_data_from_value(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|_| invalid(format!("binscatter: {context} must be numeric")))?;
    ensure_vector_tensor(&tensor, context)?;
    Ok(tensor)
}

pub(super) fn ensure_vector_tensor(tensor: &Tensor, context: &str) -> BuiltinResult<()> {
    if tensor.shape().iter().filter(|&&dim| dim > 1).count() > 1 {
        return Err(invalid(format!(
            "binscatter: {context} must be a real numeric vector"
        )));
    }
    Ok(())
}

pub(super) fn ensure_compatible_lengths(x_data: &Tensor, y_data: &Tensor) -> BuiltinResult<()> {
    if x_data.len() != y_data.len() {
        return Err(invalid(
            "binscatter: x and y must contain the same number of elements",
        ));
    }
    Ok(())
}
