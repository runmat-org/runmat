//! Metadata admission before any resident input is downloaded.
use super::{invalid_argument, parse_type, prior::PriorSpec};
use crate::BuiltinResult;
use runmat_builtins::DiscriminantType;
use runmat_value::Value;

pub(super) fn admit(
    sample: &Value,
    training: &Value,
    group: &Value,
    rest: &[Value],
) -> BuiltinResult<AdmittedHeader> {
    let sample_geometry = predictor_geometry(sample, "sample")?;
    let training_geometry = predictor_geometry(training, "training")?;
    if let (Some((_, sample_columns)), Some((_, training_columns))) =
        (sample_geometry, training_geometry)
    {
        if sample_columns != training_columns {
            return Err(invalid_argument(
                "classify: sample and training must have the same number of columns",
            ));
        }
    }
    if sample_geometry.is_some_and(|(_, columns)| columns == 0)
        || training_geometry.is_some_and(|(rows, _)| rows == 0)
    {
        return Err(invalid_argument(
            "classify: sample and training must be nonempty numeric matrices",
        ));
    }
    if let (Some((training_rows, _)), Some(group_count)) = (training_geometry, group_count(group)?)
    {
        if group_count != training_rows {
            return Err(invalid_argument(
                "classify: group length must match the number of rows in training",
            ));
        }
    }
    let discriminant = rest
        .first()
        .map_or(Ok(DiscriminantType::Linear), parse_type)?;
    let prior = PriorSpec::prepare(rest.get(1))?;
    Ok(AdmittedHeader {
        discriminant,
        prior,
    })
}

pub(super) struct AdmittedHeader {
    pub(super) discriminant: DiscriminantType,
    pub(super) prior: PriorSpec,
}

pub(super) fn preflight_numeric_vector(value: &Value, role: &str) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Err(invalid_argument(format!(
                    "classify: {role} must be a numeric vector"
                )));
            }
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) => return Ok(()),
        _ => {
            return Err(invalid_argument(format!(
                "classify: {role} must be a numeric vector"
            )))
        }
    };
    if shape.iter().filter(|dimension| **dimension > 1).count() > 1 {
        return Err(invalid_argument(format!(
            "classify: {role} must be a vector"
        )));
    }
    Ok(())
}

fn predictor_geometry(value: &Value, role: &str) -> BuiltinResult<Option<(usize, usize)>> {
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape.as_slice()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid_argument(format!(
                    "classify: {role} must be a 2-D numeric matrix"
                )));
            }
            Some(handle.shape())
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(Some((1, 1))),
        _ => None,
    };
    let Some(shape) = shape else {
        return Ok(None);
    };
    if shape.len() > 2 {
        return Err(invalid_argument(format!(
            "classify: {role} must be a 2-D numeric matrix"
        )));
    }
    Ok(Some((
        shape.first().copied().unwrap_or(1),
        shape.get(1).copied().unwrap_or(1),
    )))
}

pub(super) fn group_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let count = match value {
        Value::Tensor(tensor) => {
            if tensor
                .shape
                .iter()
                .filter(|dimension| **dimension > 1)
                .count()
                > 1
            {
                return Err(invalid_argument("classify: group must be a vector"));
            }
            Some(tensor.len())
        }
        Value::LogicalArray(array) => Some(array.data.len()),
        Value::StringArray(array) => Some(array.data.len()),
        Value::Cell(array) => {
            if array.data.iter().any(crate::dispatcher::value_contains_gpu) {
                return Err(invalid_argument(
                    "classify: group cell entries must be text",
                ));
            }
            Some(array.data.len())
        }
        Value::CharArray(array) => Some(array.rows),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid_argument(
                    "classify: group must be numeric, logical, string, character, or cellstr labels",
                ));
            }
            if handle
                .shape()
                .iter()
                .filter(|dimension| **dimension > 1)
                .count()
                > 1
            {
                return Err(invalid_argument("classify: group must be a vector"));
            }
            Some(
                handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .ok_or_else(|| invalid_argument("classify: group shape overflows"))?,
            )
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Some(1),
        _ => None,
    };
    Ok(count)
}
