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
    let shape = numeric_payload_shape(value, role.name(), function)?;
    // MATLAB's empty-value lifecycle is admitted only after the storage class
    // is known. Empty logical, sparse, complex, and non-real resident values
    // must not bypass the same class boundary as nonempty parameters.
    if shape.contains(&0) {
        return Ok(());
    }
    let valid = match expected {
        ParameterShape::Matrix { rows, columns } => match shape {
            [actual_rows, actual_columns] => {
                actual_rows == rows
                    && *actual_columns > 0
                    && columns.is_none_or(|width| *actual_columns == width)
            }
            [actual_columns] => {
                *rows == 1
                    && *actual_columns > 0
                    && columns.is_none_or(|width| *actual_columns == width)
            }
            _ => false,
        },
        ParameterShape::Dimensions(dimensions) => {
            shape.len() == dimensions.len()
                && shape.iter().zip(dimensions).all(|(actual, expected)| {
                    *actual > 0 && expected.is_none_or(|size| *actual == size)
                })
        }
        ParameterShape::Elements(expected) => {
            shape
                .iter()
                .try_fold(1usize, |size, dim| size.checked_mul(*dim))
                == Some(*expected)
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

/// Validate physical storage and ownership before interpreting geometry. This
/// is shared by parameter and recurrent-state admission, including empties.
pub(in crate::builtins::deep_learning::model) fn numeric_payload_shape<'a>(
    value: &'a Value,
    label: &str,
    function: &'static str,
) -> BuiltinResult<&'a [usize]> {
    let data = parameter_data(value, function)?;
    let shape: &[usize] = match data {
        Value::Tensor(tensor) => &tensor.shape,
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            if runmat_accelerate_api::handle_precision(handle).is_none()
                || crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle).is_none()
            {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {label} has an unsupported resident owner or class"),
                ));
            }
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|source| {
                    crate::build_runtime_error(format!(
                        "{function}: {} has a retired resident allocation",
                        label
                    ))
                    .with_builtin(function)
                    .with_identifier("RunMat:deepLearning:InvalidInput")
                    .with_source(source)
                    .build()
                })?;
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) => &[1, 1],
        _ => {
            return Err(deep_learning_error(
                function,
                format!("{function}: {} must contain real numeric storage", label),
            ))
        }
    };
    if let Value::Object(object) = value {
        if object.class_name.is(super::DLARRAY_CLASS) {
            let format = match object.properties.get("Format") {
                None => "",
                Some(Value::String(format)) => format.as_str(),
                _ => {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: {label} dlarray Format must be text"),
                    ))
                }
            };
            let labels = match object.properties.get("Labels") {
                None => format,
                Some(Value::String(labels)) => labels.as_str(),
                _ => {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: {label} dlarray Labels must be text"),
                    ))
                }
            };
            if format != labels {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {label} dlarray Format and Labels disagree"),
                ));
            }
            if !format.is_empty() {
                runmat_builtins::DeepLearningAxisLayout::parse(format, shape.len()).map_err(
                    |error| {
                        deep_learning_error(
                            function,
                            format!("{function}: {label} has invalid axis labels: {error}"),
                        )
                    },
                )?;
            }
        }
    }
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: Vec<usize>) -> Value {
        let count = shape.iter().product();
        Value::Tensor(Tensor::new(vec![1.0; count], shape).unwrap())
    }

    #[test]
    fn expanded_parameter_roles_enforce_native_geometry() {
        let convolution = ParameterShape::Dimensions(vec![Some(3), Some(2), Some(4)]);
        assert!(validate_parameter(
            &tensor(vec![3, 2, 4]),
            &convolution,
            ParameterRole::Weights,
            "test"
        )
        .is_ok());
        assert!(validate_parameter(
            &tensor(vec![3, 4, 2]),
            &convolution,
            ParameterRole::Weights,
            "test"
        )
        .is_err());

        let recurrent = ParameterShape::Matrix {
            rows: 12,
            columns: Some(2),
        };
        assert!(validate_parameter(
            &tensor(vec![12, 2]),
            &recurrent,
            ParameterRole::InputWeights,
            "test"
        )
        .is_ok());
        assert!(validate_parameter(
            &tensor(vec![12, 3]),
            &recurrent,
            ParameterRole::InputWeights,
            "test"
        )
        .is_err());

        let normalization = ParameterShape::Dimensions(vec![Some(2), Some(1)]);
        assert!(validate_parameter(
            &tensor(vec![2, 1]),
            &normalization,
            ParameterRole::Scale,
            "test"
        )
        .is_ok());
        assert!(validate_parameter(
            &tensor(vec![1, 2]),
            &normalization,
            ParameterRole::Scale,
            "test"
        )
        .is_err());
    }

    #[test]
    fn empty_logical_and_malformed_axis_metadata_never_bypass_admission() {
        let shape = ParameterShape::Elements(2);
        assert!(
            validate_parameter(&Value::Bool(false), &shape, ParameterRole::Bias, "test").is_err()
        );
        let mut wrapper = ObjectInstance::new(DLARRAY_CLASS.owned());
        wrapper.properties.insert("Data".into(), tensor(vec![2, 1]));
        wrapper
            .properties
            .insert("Format".into(), Value::String("CU".into()));
        wrapper
            .properties
            .insert("Labels".into(), Value::String("UC".into()));
        assert!(
            validate_parameter(&Value::Object(wrapper), &shape, ParameterRole::Bias, "test")
                .is_err()
        );
    }
}
