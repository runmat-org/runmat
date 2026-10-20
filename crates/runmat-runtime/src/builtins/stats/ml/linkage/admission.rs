//! Admission, resource preflight, compatibility, and numeric boundary checks for linkage.

use runmat_builtins::{
    matrix_size_for_condensed_length, LINKAGE_EXPLICIT_GPU_INPUT_EXTENSION,
    LINKAGE_INTEGER_DATA_EXTENSION, LINKAGE_INTEGER_DISTANCE_PARAMETER_EXTENSION,
};
use runmat_value::{Tensor, Value};

use super::options::{validate_savememory_for_matrix, LinkageOptions};
use super::{internal, invalid, with_linkage_context, MAX_CONDENSED_DISTANCES, NAME};
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};

pub(super) fn preflight_input_geometry(
    input: &Value,
    options: &LinkageOptions,
) -> BuiltinResult<()> {
    let shape = match input {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            return Err(invalid(
                "linkage: condensed distance vector length must be n*(n-1)/2",
            ))
        }
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid(
                    "linkage: input must be a real numeric vector or matrix",
                ));
            }
            handle.shape()
        }
        _ => return Err(invalid("linkage: input must be a numeric vector or matrix")),
    };
    let (rows, columns, vector) = match shape {
        [rows] => (*rows, 1, true),
        [rows, columns] => (*rows, *columns, *rows == 1),
        _ => {
            return Err(invalid(
                "linkage: input must be a numeric vector or 2-D matrix",
            ))
        }
    };
    rows.checked_mul(columns)
        .ok_or_else(|| internal("linkage: input shape overflow"))?;
    if vector {
        if options.metric.is_some() {
            return Err(invalid(
                "linkage: condensed distance vector input does not accept a distance metric",
            ));
        }
        if options.savememory == Some(true) {
            return Err(invalid(
                "linkage: SaveMemory 'on' is only supported for observation matrix input",
            ));
        }
        let length = if shape.len() == 1 { rows } else { columns };
        let observations = matrix_size_for_condensed_length(length)
            .filter(|observations| *observations >= 2)
            .ok_or_else(|| {
                invalid("linkage: condensed distance vector length must be n*(n-1)/2")
            })?;
        validate_observation_count(observations)
    } else {
        if rows < 2 {
            return Err(invalid(
                "linkage: X must be a numeric matrix with at least two observations in rows",
            ));
        }
        validate_savememory_for_matrix(options)?;
        validate_observation_count(rows)
    }
}

pub(super) async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(with_linkage_context)?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid(format!("linkage: {err}")))?;
    if tensor.shape().len() > 2 {
        return Err(invalid(
            "linkage: input must be a numeric vector or 2-D matrix",
        ));
    }
    ensure_exact_integer_tensor(&tensor, "X or y")?;
    Ok(tensor)
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data().iter().any(contains_typed_integer),
        Value::Struct(value) => value.fields.values().any(contains_typed_integer),
        Value::Object(value) => value.properties().values().any(contains_typed_integer),
        Value::Closure(value) => value.captures.iter().any(contains_typed_integer),
        Value::OutputList(values) => values.iter().any(contains_typed_integer),
        _ => is_typed_integer(value),
    }
}

fn contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(contains_explicit_gpu),
        Value::Object(value) => value.properties().values().any(contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(contains_explicit_gpu),
        _ => false,
    }
}

fn has_typed_integer_distance_parameter(options: &LinkageOptions) -> bool {
    options.metric.as_ref().is_some_and(|metric| {
        metric.kind.accepts_parameter() && metric.parameters.iter().any(contains_typed_integer)
    })
}

pub(super) fn ensure_extensions(input: &Value, options: &LinkageOptions) -> BuiltinResult<()> {
    if is_typed_integer(input) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LINKAGE_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    if has_typed_integer_distance_parameter(options) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LINKAGE_INTEGER_DISTANCE_PARAMETER_EXTENSION,
            NAME,
        )?;
    }
    if contains_explicit_gpu(input)
        || options
            .metric
            .as_ref()
            .is_some_and(|metric| metric.parameters.iter().any(contains_explicit_gpu))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LINKAGE_EXPLICIT_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn ensure_exact_integer_tensor(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    if storage
        .exact_values()
        .iter()
        .any(|integer| !crate::builtins::common::validation::integer_is_exact_f64(integer))
    {
        return Err(invalid(format!(
            "linkage: integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

pub(super) async fn ensure_exact_metric_integer_boundaries(values: &[Value]) -> BuiltinResult<()> {
    for value in values {
        if !contains_typed_integer(value) {
            continue;
        }
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(with_linkage_context)?;
        match gathered {
            Value::Int(integer) => {
                if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
                    return Err(invalid(
                        "linkage: integer distance parameter values must be exactly representable as double",
                    ));
                }
            }
            Value::Tensor(tensor) => {
                ensure_exact_integer_tensor(&tensor, "distance parameter")?;
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn validate_observation_count(n: usize) -> BuiltinResult<()> {
    let condensed = n
        .checked_mul(n.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal("linkage: distance count overflow"))?;
    if condensed > MAX_CONDENSED_DISTANCES {
        return Err(invalid(format!(
            "linkage: too many observations ({n}) for an in-memory clustering tree"
        )));
    }
    Ok(())
}

pub(super) fn validate_distances(values: &[f64]) -> BuiltinResult<()> {
    if values.len() > MAX_CONDENSED_DISTANCES {
        return Err(invalid(
            "linkage: condensed distance vector is too large for an in-memory clustering tree",
        ));
    }
    for value in values {
        if !value.is_finite() {
            return Err(invalid("linkage: distances must be finite"));
        }
        if *value < 0.0 {
            return Err(invalid("linkage: distances must be nonnegative"));
        }
    }
    Ok(())
}
