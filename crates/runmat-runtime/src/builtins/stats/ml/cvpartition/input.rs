use super::*;
use runmat_value::{decode_utf16_code_units, Utf16DecodeError};

#[derive(Clone, Debug)]
pub(super) struct PartitionInput {
    pub(super) n: usize,
    pub(super) labels: Option<Vec<Option<PartitionLabel>>>,
}

impl PartitionInput {
    pub(super) fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(number) => Ok(Self {
                n: positive_integer_number(number, "n")?,
                labels: None,
            }),
            Value::Int(integer) => Ok(Self {
                n: integer.try_to_usize().filter(|n| *n > 0).ok_or_else(|| {
                    invalid_argument("cvpartition: n must be a positive integer scalar")
                })?,
                labels: None,
            }),
            Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
                let n = if let Some(integer) = tensor.integer_storage().and_then(|s| s.value_at(0)) {
                    integer.try_to_usize().filter(|n| *n > 0).ok_or_else(|| {
                        invalid_argument("cvpartition: n must be a positive integer scalar")
                    })?
                } else {
                    positive_integer_number(tensor::tensor_value_f64(&tensor, 0), "n")?
                };
                Ok(Self { n, labels: None })
            }
            Value::Tensor(tensor) => numeric_labels(tensor),
            Value::LogicalArray(array) => logical_labels(array),
            Value::Bool(flag) => Ok(Self {
                n: 1,
                labels: Some(vec![Some(PartitionLabel::Logical(flag))]),
            }),
            Value::String(text) => Ok(Self {
                n: 1,
                labels: Some(vec![PartitionLabel::text(text)]),
            }),
            Value::StringArray(array) => string_array_labels(array),
            Value::CharArray(chars) => Ok(Self {
                n: chars.rows,
                labels: Some(char_row_labels(chars)?),
            }),
            other => Err(invalid_argument(format!(
                "cvpartition: first input must be an observation count or grouping variable, got {other:?}"
            ))),
        }
    }

    pub(super) fn default_stratify(&self) -> bool {
        self.labels.is_some()
    }
}

pub(super) fn numeric_labels(tensor: Tensor) -> BuiltinResult<PartitionInput> {
    let len = tensor::tensor_element_len(&tensor);
    if len == 0 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if tensor.shape().iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    let labels = if let Some(storage) = tensor.integer_storage() {
        storage
            .exact_values()
            .iter()
            .map(|value| Some(PartitionLabel::integer(value)))
            .collect()
    } else {
        tensor::tensor_values_f64_cow(&tensor)
            .iter()
            .map(|value| PartitionLabel::number(*value))
            .collect()
    };
    Ok(PartitionInput {
        n: len,
        labels: Some(labels),
    })
}

pub(super) fn logical_labels(array: LogicalArray) -> BuiltinResult<PartitionInput> {
    if array.data.is_empty() {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if array.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    Ok(PartitionInput {
        n: array.data.len(),
        labels: Some(
            array
                .data
                .into_iter()
                .map(|flag| Some(PartitionLabel::Logical(flag != 0)))
                .collect(),
        ),
    })
}

pub(super) fn string_array_labels(array: StringArray) -> BuiltinResult<PartitionInput> {
    if array.data().is_empty() {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if array.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    let (data, _shape, _rows, _cols) = array.into_parts();
    let n = data.len();
    let labels = match data.try_into_owned() {
        Ok(values) => values
            .into_iter()
            .map(|text| PartitionLabel::text(text))
            .collect(),
        Err(values) => values
            .iter()
            .cloned()
            .map(|text| PartitionLabel::text(text))
            .collect(),
    };
    Ok(PartitionInput {
        n,
        labels: Some(labels),
    })
}

pub(super) fn char_row_labels(
    chars: runmat_value::CharArray,
) -> BuiltinResult<Vec<Option<PartitionLabel>>> {
    let mut labels = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        let start = row * chars.cols;
        let label = decode_utf16_code_units(&chars.data()[start..start + chars.cols])
            .map_err(|error| invalid_utf16("stratification label", error))?;
        labels.push(Some(PartitionLabel::Text(label.into())));
    }
    Ok(labels)
}

fn invalid_utf16(context: &str, error: Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "cvpartition: {context} is not valid UTF-16 text: {error}"
    ))
    .with_builtin(CVPARTITION_NAME)
    .with_source(error);
    if let Some(identifier) = CVPARTITION_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn parse_kind(text: &str) -> BuiltinResult<PartitionKind> {
    PartitionKind::parse(text).ok_or_else(|| {
        invalid_argument(format!(
            "cvpartition: unsupported partition kind '{}'",
            canonical(text)
        ))
    })
}
