use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IncludedEdge {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GroupCountOptions {
    pub(super) include_missing: bool,
    pub(super) include_empty: bool,
    pub(super) included_edge: IncludedEdge,
}

impl Default for GroupCountOptions {
    fn default() -> Self {
        Self {
            include_missing: true,
            include_empty: false,
            included_edge: IncludedEdge::Left,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OptionName {
    IncludedEdge,
    IncludeMissingGroups,
    IncludeEmptyGroups,
}

impl GroupCountOptions {
    pub(super) fn split_and_parse(values: Vec<Value>) -> BuiltinResult<(Vec<Value>, Self)> {
        let start = option_start(crate::call::FunctionArgumentView::from_slice(&values))?;
        let (positional, pairs) = values.split_at(start);
        if !pairs.len().is_multiple_of(2) {
            return Err(error::invalid(
                "groupcounts: name-value options must be provided in pairs",
            ));
        }
        let mut options = Self::default();
        for pair in pairs.chunks_exact(2) {
            let name = OptionName::parse(&pair[0])?
                .ok_or_else(|| error::invalid("groupcounts: unsupported name-value option"))?;
            match name {
                OptionName::IncludedEdge => {
                    options.included_edge = IncludedEdge::parse(&pair[1])?;
                }
                OptionName::IncludeMissingGroups => {
                    options.include_missing = binary_boolean(&pair[1], "IncludeMissingGroups")?;
                }
                OptionName::IncludeEmptyGroups => {
                    options.include_empty = binary_boolean(&pair[1], "IncludeEmptyGroups")?;
                }
            }
        }
        Ok((positional.to_vec(), options))
    }
}

impl OptionName {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Option<Self>> {
        let Some(text) = scalar_text(value)? else {
            return Ok(None);
        };
        Ok(if text.eq_ignore_ascii_case("IncludedEdge") {
            Some(Self::IncludedEdge)
        } else if text.eq_ignore_ascii_case("IncludeMissingGroups") {
            Some(Self::IncludeMissingGroups)
        } else if text.eq_ignore_ascii_case("IncludeEmptyGroups") {
            Some(Self::IncludeEmptyGroups)
        } else {
            None
        })
    }
}

pub(super) fn option_start(values: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<usize> {
    for (index, value) in values.iter().enumerate() {
        if OptionName::parse(value)?.is_some() {
            return Ok(index);
        }
    }
    Ok(values.len())
}

pub(super) fn is_boolean_option(value: &Value) -> BuiltinResult<bool> {
    Ok(matches!(
        OptionName::parse(value)?,
        Some(OptionName::IncludeMissingGroups | OptionName::IncludeEmptyGroups)
    ))
}

impl IncludedEdge {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let value = scalar_text(value)?
            .ok_or_else(|| error::invalid("groupcounts: IncludedEdge must be 'left' or 'right'"))?;
        if value.eq_ignore_ascii_case("left") {
            Ok(Self::Left)
        } else if value.eq_ignore_ascii_case("right") {
            Ok(Self::Right)
        } else {
            Err(error::invalid(
                "groupcounts: IncludedEdge must be 'left' or 'right'",
            ))
        }
    }
}

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<Option<std::borrow::Cow<'_, str>>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(value)) => {
            Some(std::borrow::Cow::Borrowed(value.as_str()))
        }
        Value::CharArray(value) if value.rows <= 1 => Some(std::borrow::Cow::Owned(
            runmat_value::decode_utf16_code_units(value.data()).map_err(|source| {
                let mut error =
                    error::invalid(format!("groupcounts: invalid option text: {source}"));
                error.source = Some(Box::new(source));
                error
            })?,
        )),
        _ => None,
    })
}

fn binary_boolean(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value
            .try_to_usize()
            .filter(|value| *value <= 1)
            .map(|value| value == 1)
            .ok_or_else(|| error::invalid(format!("groupcounts: {name} must be 0 or 1")));
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value == 1.0),
        Value::Tensor(value) if value.len() == 1 => {
            let value = value.materialize_f64()[0];
            if value == 0.0 || value == 1.0 {
                Ok(value == 1.0)
            } else {
                Err(error::invalid(format!(
                    "groupcounts: {name} must be 0 or 1"
                )))
            }
        }
        _ => Err(error::invalid(format!(
            "groupcounts: {name} must be a logical or numeric scalar 0 or 1"
        ))),
    }
}
