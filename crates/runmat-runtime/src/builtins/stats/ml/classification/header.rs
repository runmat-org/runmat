//! Metadata admission before any resident input is downloaded.
use super::{
    ensure_exact_integer_scalar_boundary, gathered, invalid_argument, is_missing_label,
    labels_from_value, parse_type, prior::PriorSpec, same_label, LabelVector,
};
use crate::BuiltinResult;
use runmat_builtins::DiscriminantType;
use runmat_value::Value;

pub(super) fn admit(
    sample: &Value,
    training: &Value,
    group: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<AdmittedHeader> {
    let sample_geometry = predictor_geometry(sample, "sample")?;
    let training_geometry = predictor_geometry(training, "training")?;
    if sample_geometry.1 != training_geometry.1 {
        return Err(invalid_argument(
            "classify: sample and training must have the same number of columns",
        ));
    }
    if sample_geometry.1 == 0 || training_geometry.0 == 0 {
        return Err(invalid_argument(
            "classify: sample and training must be nonempty numeric matrices",
        ));
    }
    preflight_host_predictor(sample, "sample")?;
    preflight_host_predictor(training, "training")?;
    let group = GroupSpec::prepare(group)?;
    if let Some(group_count) = group.known_len() {
        if group_count != training_geometry.0 {
            return Err(invalid_argument(
                "classify: group length must match the number of rows in training",
            ));
        }
    }
    if let GroupSpec::Host(labels) = &group {
        if !has_two_nonmissing_classes(labels) {
            return Err(invalid_argument(
                "classify: group must contain at least two nonmissing classes",
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
        group,
    })
}

pub(super) struct AdmittedHeader {
    pub(super) discriminant: DiscriminantType,
    pub(super) prior: PriorSpec,
    pub(super) group: GroupSpec,
}

pub(super) enum GroupSpec {
    Host(LabelVector),
    Resident(Value),
}

impl GroupSpec {
    fn prepare(value: Value) -> BuiltinResult<Self> {
        let count = group_count(&value)?;
        if matches!(value, Value::GpuTensor(_)) {
            return Ok(Self::Resident(value));
        }
        let labels = labels_from_value(value, "group")?;
        debug_assert_eq!(count, Some(labels.labels.len()));
        Ok(Self::Host(labels))
    }

    fn known_len(&self) -> Option<usize> {
        match self {
            Self::Host(labels) => Some(labels.labels.len()),
            Self::Resident(value) => group_count(value).ok().flatten(),
        }
    }

    pub(super) async fn materialize(self) -> BuiltinResult<LabelVector> {
        match self {
            Self::Host(labels) => Ok(labels),
            Self::Resident(value) => labels_from_value(gathered(value).await?, "group"),
        }
    }
}

fn has_two_nonmissing_classes(group: &LabelVector) -> bool {
    let mut first = None;
    for label in &group.labels {
        if is_missing_label(label) {
            continue;
        }
        match first {
            None => first = Some(label),
            Some(existing) if !same_label(existing, label) => return true,
            Some(_) => {}
        }
    }
    false
}

fn preflight_host_predictor(value: &Value, role: &str) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) => {
            for index in 0..tensor.len() {
                let scalar = tensor.numeric_value_at(index).ok_or_else(|| {
                    invalid_argument(format!("classify: {role} numeric storage is invalid"))
                })?;
                ensure_exact_integer_scalar_boundary(scalar, role)?;
                if !scalar.is_finite() {
                    return Err(invalid_argument(
                        "classify: sample and training must contain finite values",
                    ));
                }
            }
        }
        Value::Num(number) if !number.is_finite() => {
            return Err(invalid_argument(
                "classify: sample and training must contain finite values",
            ));
        }
        Value::Int(integer) => {
            ensure_exact_integer_scalar_boundary(
                runmat_value::NumericScalar::from(integer.clone()),
                role,
            )?;
        }
        Value::GpuTensor(_) | Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_) => {}
        _ => {
            return Err(invalid_argument(format!(
                "classify: {role} must be a 2-D numeric matrix"
            )));
        }
    }
    Ok(())
}

pub(super) fn preflight_numeric_vector(value: &Value, role: &str) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
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

fn predictor_geometry(value: &Value, role: &str) -> BuiltinResult<(usize, usize)> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid_argument(format!(
                    "classify: {role} must be a 2-D numeric matrix"
                )));
            }
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok((1, 1)),
        _ => {
            return Err(invalid_argument(format!(
                "classify: {role} must be a 2-D numeric matrix"
            )));
        }
    };
    if shape.len() > 2 {
        return Err(invalid_argument(format!(
            "classify: {role} must be a 2-D numeric matrix"
        )));
    }
    Ok((
        shape.first().copied().unwrap_or(1),
        shape.get(1).copied().unwrap_or(1),
    ))
}

pub(super) fn group_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let count = match value {
        Value::Tensor(tensor) => {
            ensure_group_vector(tensor.shape())?;
            Some(tensor.len())
        }
        Value::LogicalArray(array) => {
            ensure_group_vector(&array.shape)?;
            Some(array.data.len())
        }
        Value::StringArray(array) => {
            ensure_group_vector(&array.shape)?;
            Some(array.data().len())
        }
        Value::Cell(array) => {
            ensure_group_vector(&array.shape)?;
            if array
                .data()
                .iter()
                .any(crate::dispatcher::value_contains_gpu)
            {
                return Err(invalid_argument(
                    "classify: group cell entries must be text",
                ));
            }
            Some(array.data().len())
        }
        Value::CharArray(array) => Some(array.rows),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid_argument(
                    "classify: group must be numeric, logical, string, character, or cellstr labels",
                ));
            }
            ensure_group_vector(handle.shape())?;
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

fn ensure_group_vector(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().filter(|dimension| **dimension > 1).count() > 1 {
        return Err(invalid_argument("classify: group must be a vector"));
    }
    Ok(())
}
