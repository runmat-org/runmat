use runmat_value::{NumericScalar, Value};

use crate::BuiltinResult;

use super::super::super::{
    deep_learning_error, logical_scalar, numeric_scalar, positive_usize, scalar_text,
};

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

#[derive(Clone, Copy)]
pub(super) enum PaddingDirection {
    Left,
    Right,
    Both,
}

impl PaddingDirection {
    pub(super) fn parse(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            "both" => Ok(Self::Both),
            other => Err(deep_learning_error(
                "padsequences",
                format!(
                    "padsequences: PaddingDirection must be 'left', 'right', or 'both', got '{other}'"
                ),
            )),
        }
    }
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
                match scalar_text(value, "padsequences")?.to_ascii_lowercase().as_str() {
                    "longest" => Ok(Self::Longest),
                    "shortest" => Ok(Self::Shortest),
                    other => Err(deep_learning_error(
                        "padsequences",
                        format!("padsequences: Length must be 'longest', 'shortest', or a positive integer, got '{other}'"),
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
            match scalar_text(value, function)?.to_ascii_lowercase().as_str() {
                "true" | "on" | "yes" => Ok(true),
                "false" | "off" | "no" => Ok(false),
                other => Err(deep_learning_error(
                    function,
                    format!("{function}: {label} must be logical, got '{other}'"),
                )),
            }
        }
        _ => logical_scalar(value, function, label).map_err(|_| {
            deep_learning_error(function, format!("{function}: {label} must be logical"))
        }),
    }
}
