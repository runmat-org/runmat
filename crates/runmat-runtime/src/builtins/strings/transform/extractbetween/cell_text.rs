use crate::builtins::strings::common::char_row_to_string_slice;
use runmat_value::{CharArray, Value};

#[derive(Clone, Copy, Debug)]
pub(super) enum CellElementKind {
    String,
    Char,
}

/// A scalar text element inside a cell array. `Value::Cell` itself determines
/// the container's identity; the element check only validates its text payload.
pub(super) enum TextCellScalar<'a> {
    String(&'a runmat_value::StringElement),
    Char(&'a CharArray),
}

impl<'a> TextCellScalar<'a> {
    pub(super) fn from_value(value: &'a Value) -> Option<Self> {
        match value {
            Value::String(text) => Some(Self::String(text)),
            Value::StringArray(array) if array.data().len() == 1 => {
                Some(Self::String(&array.data()[0]))
            }
            Value::CharArray(array) if array.rows <= 1 => Some(Self::Char(array)),
            _ => None,
        }
    }

    pub(super) fn text(
        &self,
    ) -> Result<runmat_value::StringElement, runmat_value::Utf16DecodeError> {
        match self {
            Self::String(text) => Ok((*text).clone()),
            Self::Char(array) if array.rows == 0 => Ok(runmat_value::StringElement::text("")),
            Self::Char(array) => char_row_to_string_slice(array.data(), array.cols, 0)
                .map(runmat_value::StringElement::text),
        }
    }

    pub(super) fn kind(&self) -> CellElementKind {
        match self {
            Self::String(_) => CellElementKind::String,
            Self::Char(_) => CellElementKind::Char,
        }
    }
}
