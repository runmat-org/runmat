use runmat_builtins::{PadBooleanKeyword, PadLengthKeyword, PadOptionKey};
use runmat_value::{NumericScalar, Value};
use std::collections::BTreeMap;

use crate::BuiltinResult;

use super::super::super::{
    deep_learning_error, logical_scalar, numeric_scalar, positive_usize, scalar_text,
};
pub(super) use runmat_builtins::PadDirection as PaddingDirection;

pub(super) enum AdmittedPadOption {
    PaddingValue(NumericScalar),
    Direction(PaddingDirection),
    Length(SequenceLength),
    UniformOutput(bool),
    Resident(Value),
}

pub(super) type AdmittedPadOptions = BTreeMap<PadOptionKey, AdmittedPadOption>;

pub(super) fn numeric_pad_value(value: &Value) -> BuiltinResult<NumericScalar> {
    numeric_scalar(value, "padsequences", "PaddingValue")?;
    match value {
        Value::Num(value) => Ok(NumericScalar::F64(*value)),
        Value::Int(value) => Ok(NumericScalar::from(value.clone())),
        Value::Tensor(value) => value.numeric_value_at(0).ok_or_else(|| {
            deep_learning_error("padsequences", "padsequences: PaddingValue is empty")
        }),
        _ => unreachable!("numeric scalar admitted by validator"),
    }
}

#[derive(Clone, Copy)]
pub(super) struct PadOptions {
    pub(super) padding_value: NumericScalar,
    pub(super) direction: PaddingDirection,
    pub(super) length: SequenceLength,
    pub(super) uniform_output: bool,
}

impl Default for PadOptions {
    fn default() -> Self {
        Self {
            padding_value: NumericScalar::F64(0.0),
            direction: PaddingDirection::Right,
            length: SequenceLength::Longest,
            uniform_output: true,
        }
    }
}

pub(super) fn parse_direction(value: &Value) -> BuiltinResult<PaddingDirection> {
    let text = scalar_text(value, "padsequences")?;
    PaddingDirection::parse(&text).ok_or_else(|| {
        deep_learning_error(
            "padsequences",
            format!(
                "padsequences: PaddingDirection must be 'left', 'right', or 'both', got '{}'",
                text.to_ascii_lowercase()
            ),
        )
    })
}

#[derive(Clone, Copy)]
pub(in crate::builtins::deep_learning) enum SequenceLength {
    Longest,
    Shortest,
    Fixed(usize),
}

impl SequenceLength {
    pub(in crate::builtins::deep_learning) fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
                let text = scalar_text(value, "padsequences")?;
                match PadLengthKeyword::parse(&text) {
                    Some(PadLengthKeyword::Longest) => Ok(Self::Longest),
                    Some(PadLengthKeyword::Shortest) => Ok(Self::Shortest),
                    None => Err(deep_learning_error(
                        "padsequences",
                        format!("padsequences: Length must be 'longest', 'shortest', or a positive integer, got '{}'", text.to_ascii_lowercase()),
                    )),
                }
            }
            _ => positive_usize(value, "padsequences", "Length").map(Self::Fixed),
        }
    }
}

pub(super) fn parse_bool(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<bool> {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let text = scalar_text(value, function)?;
            match PadBooleanKeyword::parse(&text) {
                Some(keyword) => Ok(keyword.value()),
                None => Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: {label} must be logical, got '{}'",
                        text.to_ascii_lowercase()
                    ),
                )),
            }
        }
        _ => logical_scalar(value, function, label).map_err(|_| {
            deep_learning_error(function, format!("{function}: {label} must be logical"))
        }),
    }
}
