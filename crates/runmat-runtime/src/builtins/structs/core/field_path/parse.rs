use super::{FieldPath, FieldStep, IndexComponent, IndexSelector, PathError, PathErrorKind};
use crate::builtins::common::tensor;
use runmat_value::{NumericScalar, Value};

pub(in crate::builtins::structs::core) fn parse(
    arguments: Vec<Value>,
    builtin: &str,
) -> Result<FieldPath, PathError> {
    if arguments.is_empty() {
        return Err(error(
            PathErrorKind::MissingPath,
            "expected at least one field name",
        ));
    }
    let mut arguments = arguments.into_iter().peekable();
    let mut path = FieldPath::default();
    if arguments.peek().is_some_and(is_selector) {
        path.leading_index = Some(parse_selector(
            arguments.next().expect("peeked value"),
            builtin,
            &mut path.uses_textual_index,
        )?);
    }
    while let Some(argument) = arguments.next() {
        let name = parse_field_name(argument)?;
        let index = if arguments.peek().is_some_and(is_selector) {
            Some(parse_selector(
                arguments.next().expect("peeked value"),
                builtin,
                &mut path.uses_textual_index,
            )?)
        } else {
            None
        };
        path.fields.push(FieldStep { name, index });
    }
    if path.fields.is_empty() {
        return Err(error(
            PathErrorKind::MissingPath,
            "expected field name after indices",
        ));
    }
    Ok(path)
}

fn is_selector(value: &Value) -> bool {
    matches!(value, Value::Cell(_))
}

fn parse_selector(
    value: Value,
    builtin: &str,
    uses_textual_index: &mut bool,
) -> Result<IndexSelector, PathError> {
    let Value::Cell(cell) = value else {
        unreachable!("selector shape checked by caller")
    };
    if cell.data.is_empty() {
        return Err(error(
            PathErrorKind::EmptySelector,
            "index cell must contain at least one element",
        ));
    }
    let components = cell
        .iter_column_major()
        .map(|value| parse_component(value, builtin, uses_textual_index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IndexSelector { components })
}

fn parse_component(
    value: &Value,
    builtin: &str,
    uses_textual_index: &mut bool,
) -> Result<IndexComponent, PathError> {
    match value {
        Value::Bool(value) => Ok(IndexComponent::Logical(vec![u8::from(*value)])),
        Value::LogicalArray(array) => Ok(IndexComponent::Logical(array.data.to_vec())),
        Value::CharArray(array) => {
            parse_text(&array.data.iter().collect::<String>(), uses_textual_index)
        }
        Value::String(text) => parse_text(text, uses_textual_index),
        Value::StringArray(array) if array.data.len() == 1 => {
            parse_text(&array.data[0], uses_textual_index)
        }
        Value::Tensor(array) if array.len() != 1 => {
            let indices = if let Some(result) =
                tensor::integer_tensor_dimension_vector(array, builtin, false)
            {
                result.map_err(|detail| error(PathErrorKind::InvalidIndex, detail))?
            } else {
                (0..array.len())
                    .map(|index| {
                        array
                            .numeric_value_at(index)
                            .ok_or_else(|| {
                                error(
                                    PathErrorKind::InvalidIndex,
                                    "numeric index storage is invalid",
                                )
                            })
                            .and_then(parse_numeric_scalar)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            Ok(IndexComponent::Vector(indices, array.shape.clone()))
        }
        _ => parse_scalar(value).map(IndexComponent::Scalar),
    }
}

fn parse_text(text: &str, uses_textual_index: &mut bool) -> Result<IndexComponent, PathError> {
    *uses_textual_index = true;
    let text = text.trim();
    if text.eq_ignore_ascii_case("end") {
        return Ok(IndexComponent::End);
    }
    if let Ok(index) = text.parse::<usize>() {
        return (index > 0)
            .then_some(IndexComponent::Scalar(index))
            .ok_or_else(|| error(PathErrorKind::InvalidIndex, "index must be at least one"));
    }
    Err(error(
        PathErrorKind::InvalidIndex,
        format!("invalid textual index '{text}'"),
    ))
}

fn parse_scalar(value: &Value) -> Result<usize, PathError> {
    match value {
        Value::Int(value) => value
            .try_to_usize()
            .filter(|index| *index > 0)
            .ok_or_else(|| error(PathErrorKind::InvalidIndex, "index must be at least one")),
        Value::Tensor(array) if tensor::is_scalar_tensor(array) => array
            .numeric_value_at(0)
            .ok_or_else(|| {
                error(
                    PathErrorKind::InvalidIndex,
                    "numeric index storage is invalid",
                )
            })
            .and_then(parse_numeric_scalar),
        Value::Num(value) => parse_float(*value),
        other => Err(error(
            PathErrorKind::InvalidIndex,
            format!("expected a positive integer index, got {other:?}"),
        )),
    }
}

fn parse_numeric_scalar(value: NumericScalar) -> Result<usize, PathError> {
    match value {
        NumericScalar::F64(value) => parse_float(value),
        NumericScalar::F32(value) => parse_float(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_usize())
            .filter(|index| *index > 0)
            .ok_or_else(|| error(PathErrorKind::InvalidIndex, "index must be at least one")),
    }
}

pub(super) fn parse_float(value: f64) -> Result<usize, PathError> {
    if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value >= usize::MAX as f64 {
        return Err(error(
            PathErrorKind::InvalidIndex,
            "index must be a finite positive integer within platform limits",
        ));
    }
    Ok(value as usize)
}

fn parse_field_name(value: Value) -> Result<String, PathError> {
    match value {
        Value::String(value) => Ok(value),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(error(
            PathErrorKind::FieldName,
            "field names must be scalar strings or character row vectors",
        )),
    }
}

fn error(kind: PathErrorKind, detail: impl Into<String>) -> PathError {
    PathError {
        kind,
        detail: detail.into(),
    }
}
