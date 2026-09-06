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
            Value::String(text) => Ok(Self::scalar(TextRepresentation::String, text.clone())),
            Value::StringArray(array) => Ok(Self {
                representation: TextRepresentation::StringArray,
                values: array.data.clone(),
                shape: array.shape.clone(),
            }),
            Value::CharArray(array) if array.rows == 1 => Ok(Self::scalar(
                TextRepresentation::Character,
                array.data.iter().take(array.cols).collect(),
            )),
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
        let mut values = Vec::with_capacity(array.data.len());
        for value in &array.data {
            let Value::CharArray(chars) = value else {
                return Err(super::error::catalog(error, identity));
            };
            if chars.rows != 1 {
                return Err(super::error::catalog(error, identity));
            }
            values.push(chars.data.iter().take(chars.cols).collect());
        }
        Ok(Self {
            representation: TextRepresentation::Cell,
            values,
            shape: array.shape.clone(),
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
            Ok(Value::String(values.into_iter().next().unwrap_or_default()))
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
            CellArray::new_with_shape(cells, shape.to_vec())
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
