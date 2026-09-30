//! Storage, shape, and format admission before parameter publication.
use super::*;

pub(super) fn parameter_data<'a>(
    value: &'a Value,
    function: &'static str,
) -> BuiltinResult<&'a Value> {
    match value {
        Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS) => {
            object.properties.get("Data").ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: dlarray parameter is missing Data"),
                )
            })
        }
        other => Ok(other),
    }
}

pub(super) fn empty_parameter(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.shape.contains(&0),
        Value::GpuTensor(handle) => handle.shape().contains(&0),
        _ => false,
    }
}

pub(super) fn validate_parameter(
    value: &Value,
    expected: &ParameterShape,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<()> {
    let data = parameter_data(value, function)?;
    let shape: &[usize] = match data {
        Value::Tensor(tensor) => &tensor.shape,
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) => &[1, 1],
        _ => {
            return Err(deep_learning_error(
                function,
                format!(
                    "{function}: {} must contain real numeric storage",
                    role.name()
                ),
            ))
        }
    };
    // MATLAB's empty-value lifecycle is admitted only after the storage class
    // is known. Empty logical, sparse, complex, and non-real resident values
    // must not bypass the same class boundary as nonempty parameters.
    if shape.contains(&0) {
        return Ok(());
    }
    let valid = match role {
        ParameterRole::Weights => match shape {
            [rows, columns] => {
                *rows == expected.rows
                    && *columns > 0
                    && expected.columns.is_none_or(|width| *columns == width)
            }
            [columns] => {
                expected.rows == 1
                    && *columns > 0
                    && expected.columns.is_none_or(|width| *columns == width)
            }
            _ => false,
        },
        ParameterRole::Bias => {
            shape
                .iter()
                .try_fold(1usize, |size, dim| size.checked_mul(*dim))
                == Some(expected.rows)
        }
    };
    if !valid {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: {} has shape {shape:?}, expected {expected:?}",
                role.name()
            ),
        ));
    }
    Ok(())
}
