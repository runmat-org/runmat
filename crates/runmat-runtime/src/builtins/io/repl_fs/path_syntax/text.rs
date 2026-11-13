use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::BuiltinResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TextRepresentation {
    Character,
    String,
    StringArray,
    Cell,
}

#[derive(Debug)]
pub(super) struct TextContainer {
    pub representation: TextRepresentation,
    pub values: Vec<String>,
    pub shape: Vec<usize>,
}

impl TextContainer {
    pub fn decode(
        value: &Value,
        identity: &'static str,
        error: &'static runmat_builtins::BuiltinErrorDescriptor,
    ) -> BuiltinResult<Self> {
        match value {
            Value::String(runmat_value::StringElement::Text(text)) => Ok(Self::scalar(
                TextRepresentation::String,
                text.as_str().to_owned(),
            )),
            Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
                Ok(Self {
                    representation: TextRepresentation::StringArray,
                    values: array
                        .data()
                        .iter()
                        .map(|text| text.as_str().expect("checked nonmissing string").to_owned())
                        .collect(),
                    shape: array.shape.as_slice().to_vec(),
                })
            }
            Value::CharArray(array)
                if array.rows == 1
                    && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
            {
                crate::builtins::io::utf16::decode_utf16_with_error(
                    &array.data()[..array.cols],
                    || super::error::catalog(error, identity),
                )
                .map(|text| Self::scalar(TextRepresentation::Character, text))
            }
            Value::Cell(array) => Self::decode_cell(array, identity, error),
            _ => Err(super::error::catalog(error, identity)),
        }
    }

    fn scalar(representation: TextRepresentation, value: String) -> Self {
        Self {
            representation,
            values: vec![value],
            shape: vec![1, 1],
        }
    }

    fn decode_cell(
        array: &CellArray,
        identity: &'static str,
        error: &'static runmat_builtins::BuiltinErrorDescriptor,
    ) -> BuiltinResult<Self> {
        let mut values = Vec::with_capacity(array.data().len());
        for value in array.iter_column_major() {
            let Value::CharArray(chars) = value else {
                return Err(super::error::catalog(error, identity));
            };
            if chars.rows != 1 || crate::builtins::common::shape::effective_rank(&chars.shape) > 2 {
                return Err(super::error::catalog(error, identity));
            }
            values.push(crate::builtins::io::utf16::decode_utf16_with_error(
                &chars.data()[..chars.cols],
                || super::error::catalog(error, identity),
            )?);
        }
        Ok(Self {
            representation: TextRepresentation::Cell,
            values,
            shape: array.shape.as_slice().to_vec(),
        })
    }

    pub fn is_scalar(&self) -> bool {
        self.values.len() == 1
    }
    pub fn value_at(&self, index: usize) -> &str {
        if self.is_scalar() {
            &self.values[0]
        } else {
            &self.values[index]
        }
    }
}

pub(super) fn output(
    representation: TextRepresentation,
    values: Vec<String>,
    shape: &[usize],
    identity: &'static str,
    shape_error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    match representation {
        TextRepresentation::Character => Ok(Value::CharArray(CharArray::new_row(
            values.first().map_or("", String::as_str),
        ))),
        TextRepresentation::String if values.len() == 1 => {
            Ok(Value::string(values.into_iter().next().unwrap_or_default()))
        }
        TextRepresentation::String | TextRepresentation::StringArray => {
            StringArray::new(values, shape.to_vec())
                .map(Value::StringArray)
                .map_err(|message| super::error::detail(shape_error, identity, message))
        }
        TextRepresentation::Cell => {
            let cells = values
                .into_iter()
                .map(|text| Value::CharArray(CharArray::new_row(&text)))
                .collect();
            CellArray::from_column_major(cells, shape.to_vec())
                .map(Value::Cell)
                .map_err(|message| super::error::detail(shape_error, identity, message))
        }
    }
}

pub(super) fn target_shape(arguments: &[TextContainer]) -> Result<Vec<usize>, ()> {
    let mut selected: Option<&[usize]> = None;
    for argument in arguments.iter().filter(|argument| !argument.is_scalar()) {
        match selected {
            None => selected = Some(&argument.shape),
            Some(shape) if shape != argument.shape => return Err(()),
            Some(_) => {}
        }
    }
    Ok(selected.unwrap_or(&[1, 1]).to_vec())
}

pub(super) fn result_representation(arguments: &[TextContainer]) -> TextRepresentation {
    if arguments.iter().any(|argument| {
        matches!(
            argument.representation,
            TextRepresentation::String | TextRepresentation::StringArray
        )
    }) {
        if arguments
            .iter()
            .any(|argument| argument.representation == TextRepresentation::StringArray)
        {
            TextRepresentation::StringArray
        } else {
            TextRepresentation::String
        }
    } else if arguments
        .iter()
        .any(|argument| argument.representation == TextRepresentation::Cell)
    {
        TextRepresentation::Cell
    } else {
        TextRepresentation::Character
    }
}
