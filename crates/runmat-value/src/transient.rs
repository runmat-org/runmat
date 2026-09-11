use std::fmt;

use crate::Value;

/// A structural violation caused by embedding an execution-only carrier in a
/// language value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransientValueError {
    LegacyOutputList,
}

impl fmt::Display for TransientValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyOutputList => {
                formatter.write_str("a language value contains a legacy output-list carrier")
            }
        }
    }
}

impl std::error::Error for TransientValueError {}

/// Verifies that `value` does not contain an execution-only output carrier.
///
/// The check is recursive so aggregate construction cannot hide a transient
/// sequence from workspace, persistence, foreign, or executor boundaries.
pub fn validate_no_transient_sequence(value: &Value) -> Result<(), TransientValueError> {
    match value {
        Value::OutputList(_) => Err(TransientValueError::LegacyOutputList),
        Value::Cell(cell) => validate_values(&cell.data),
        Value::Struct(structure) => validate_values(structure.fields.values()),
        Value::StructArray(array) => {
            let mut result = Ok(());
            array.for_each_value(|value| {
                if result.is_ok() {
                    result = validate_no_transient_sequence(value);
                }
            });
            result
        }
        Value::Object(object) => validate_values(object.properties.values()),
        Value::ObjectArray(array) => validate_values(array.data()),
        Value::Closure(closure) => validate_values(&closure.captures),
        _ => Ok(()),
    }
}

fn validate_values<'a>(
    values: impl IntoIterator<Item = &'a Value>,
) -> Result<(), TransientValueError> {
    for value in values {
        validate_no_transient_sequence(value)?;
    }
    Ok(())
}
