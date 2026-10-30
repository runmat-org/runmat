//! Borrowed `assignin` selectors and names. The selected workspace operation
//! admits any retained name; decoding its scalar text must not first allocate
//! a detached copy outside that operation.

use runmat_value::Value;

use crate::builtins::common::identifiers::{is_valid_varname, MATLAB_NAME_LENGTH_MAX};
use crate::RuntimeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AssigninSelector {
    Caller,
    Base,
    Unknown,
}

pub(super) fn selector(value: &Value) -> Result<AssigninSelector, RuntimeError> {
    let selected = match value {
        Value::String(text) => classify(text.as_str().ok_or_else(argument_type_error)?),
        Value::StringArray(array) if array.data().len() == 1 => {
            classify(array.data()[0].as_str().ok_or_else(argument_type_error)?)
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            if char_row_eq_ascii(chars.data(), b"caller") {
                AssigninSelector::Caller
            } else if char_row_eq_ascii(chars.data(), b"base") {
                AssigninSelector::Base
            } else {
                AssigninSelector::Unknown
            }
        }
        _ => return Err(argument_type_error()),
    };
    Ok(selected)
}

pub(super) fn name<'a>(
    value: &'a Value,
    character_row: &'a mut [u8; MATLAB_NAME_LENGTH_MAX],
) -> Result<&'a str, RuntimeError> {
    let name = match value {
        Value::String(text) => text.as_str().ok_or_else(invalid_name_error)?,
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().ok_or_else(invalid_name_error)?
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            if chars.data().len() > character_row.len() {
                return Err(invalid_name_error());
            }
            for (byte, character) in character_row.iter_mut().zip(chars.data().iter()) {
                if !character.is_ascii() {
                    return Err(invalid_name_error());
                }
                *byte = character.raw() as u8;
            }
            std::str::from_utf8(&character_row[..chars.data().len()])
                .map_err(|_| invalid_name_error())?
        }
        _ => return Err(argument_type_error()),
    };
    if !is_valid_varname(name) {
        return Err(invalid_name_error());
    }
    Ok(name)
}

fn classify(text: &str) -> AssigninSelector {
    if text.eq_ignore_ascii_case("caller") {
        AssigninSelector::Caller
    } else if text.eq_ignore_ascii_case("base") {
        AssigninSelector::Base
    } else {
        AssigninSelector::Unknown
    }
}

fn char_row_eq_ascii(row: &[runmat_value::Utf16CodeUnit], expected: &[u8]) -> bool {
    row.len() == expected.len()
        && row.iter().zip(expected).all(|(character, byte)| {
            character.is_ascii() && (character.raw() as u8).to_ascii_lowercase() == *byte
        })
}

fn argument_type_error() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:DynamicWorkspaceArgumentType",
        "assignin expects string scalar or character-row selectors",
    )
}

fn invalid_name_error() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:DynamicWorkspaceName",
        runmat_builtins::ASSIGNIN_ERROR_INVALID_NAME.message,
    )
}
