mod cell;
mod character;
mod input;
mod string;
mod symbolic;

use runmat_builtins::{CELLSTR_CELL_INPUT_EXTENSION, CELLSTR_SYMBOLIC_INPUT_EXTENSION};
use runmat_value::Value;

pub(super) fn convert(value: Value) -> crate::BuiltinResult<Value> {
    match input::classify(value) {
        input::Input::Character(array) => character::convert(array),
        input::Input::String(text) => string::scalar(text),
        input::Input::StringArray(array) => string::array(array),
        input::Input::Cell(array) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CELLSTR_CELL_INPUT_EXTENSION,
                "cellstr",
            )?;
            cell::convert(array)
        }
        input::Input::Symbolic(expression) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CELLSTR_SYMBOLIC_INPUT_EXTENSION,
                "cellstr",
            )?;
            symbolic::scalar(expression)
        }
        input::Input::SymbolicArray(array) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CELLSTR_SYMBOLIC_INPUT_EXTENSION,
                "cellstr",
            )?;
            symbolic::array(array)
        }
        input::Input::Unsupported => Err(super::error::invalid_input()),
    }
}
