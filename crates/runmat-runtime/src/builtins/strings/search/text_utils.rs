//! Shared helpers for string search builtins (contains, startsWith, etc.).

use std::convert::TryFrom;

use runmat_value::{
    CellArray, CharArray, LogicalArray, StringArray, StringElement, Utf8Text, Value,
};

use crate::builtins::common::tensor;
use crate::builtins::strings::common::char_row_to_string;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone)]
pub(crate) struct TextCollection {
    pub(crate) elements: Vec<TextElement>,
    pub(crate) shape: Vec<usize>,
    pub(crate) is_cell: bool,
}

#[derive(Clone)]
pub(crate) enum TextElement {
    Missing,
    Text(Utf8Text),
}

impl TextCollection {
    pub(crate) fn from_subject(fn_name: &str, value: Value) -> BuiltinResult<Self> {
        Self::from_argument(fn_name, value, "first argument")
    }

    pub(crate) fn from_pattern(fn_name: &str, value: Value) -> BuiltinResult<Self> {
        Self::from_argument(fn_name, value, "pattern")
    }

    pub(crate) fn from_argument(
        fn_name: &str,
        value: Value,
        descriptor: &str,
    ) -> BuiltinResult<Self> {
        Self::from_value_internal(fn_name, value, descriptor)
    }

    fn from_value_internal(fn_name: &str, value: Value, descriptor: &str) -> BuiltinResult<Self> {
        let collection = match value {
            Value::StringArray(array) => Ok(Self::from_string_array(array)),
            Value::String(text) => Ok(Self::from_string_scalar(text)),
            Value::CharArray(array) => Self::from_char_array(fn_name, array),
            Value::Cell(cell) => Self::from_cell_array(fn_name, cell),
            _ => Err(text_error(
                fn_name,
                format!(
                    "{fn_name}: {descriptor} must be text (string array, character array, or cell array of character vectors)"
                ),
            )),
        }?;

        if collection.elements.is_empty()
            || tensor::element_count(&collection.shape) == collection.elements.len()
        {
            Ok(collection)
        } else {
            Err(text_error(
                fn_name,
                format!(
                    "{fn_name}: {descriptor} must be text (string array, character array, or cell array of character vectors)"
                ),
            ))
        }
    }

    fn from_string_array(array: StringArray) -> Self {
        let (data, shape, _, _) = array.into_parts();
        let elements = match data.try_into_owned() {
            Ok(values) => values.into_iter().map(make_text_element).collect(),
            Err(data) => data.iter().cloned().map(make_text_element).collect(),
        };
        Self {
            elements,
            shape: shape.to_vec(),
            is_cell: false,
        }
    }

    fn from_string_scalar(text: StringElement) -> Self {
        Self {
            elements: vec![make_text_element(text)],
            shape: vec![1, 1],
            is_cell: false,
        }
    }

    fn from_char_array(fn_name: &str, array: CharArray) -> BuiltinResult<Self> {
        if array.rows == 0 {
            return Ok(Self {
                elements: Vec::new(),
                shape: vec![0, 1],
                is_cell: false,
            });
        }
        if array.rows == 1 {
            let text = char_row_to_string(&array, 0)
                .map_err(|source| text_decode_error(fn_name, source))?;
            return Ok(Self {
                elements: vec![TextElement::Text(text.into())],
                shape: vec![1, 1],
                is_cell: false,
            });
        }
        let mut elements = Vec::with_capacity(array.rows);
        for row in 0..array.rows {
            let text = char_row_to_string(&array, row)
                .map_err(|source| text_decode_error(fn_name, source))?;
            elements.push(TextElement::Text(text.into()));
        }
        Ok(Self {
            elements,
            shape: vec![array.rows, 1],
            is_cell: false,
        })
    }

    fn from_cell_array(fn_name: &str, cell: CellArray) -> BuiltinResult<Self> {
        let (data, _, rows, cols) = cell.into_parts();
        let mut elements = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                let idx = row * cols + col;
                let value = &data[idx];
                let element = cell_value_to_text(fn_name, value)?;
                elements.push(element);
            }
        }
        Ok(Self {
            elements,
            shape: vec![rows, cols],
            is_cell: true,
        })
    }

    pub(crate) fn lowercased(&self) -> Vec<Option<String>> {
        self.elements
            .iter()
            .map(|element| match element {
                TextElement::Missing => None,
                TextElement::Text(text) => Some(text.to_lowercase()),
            })
            .collect()
    }
}

pub(crate) fn parse_ignore_case(
    fn_name: &str,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<bool> {
    if rest.is_empty() {
        return Ok(false);
    }

    if rest.len() == 1 {
        if let Some(name) = value_to_owned_string(&rest[0], fn_name)? {
            if name.eq_ignore_ascii_case("ignorecase") {
                return Err(text_error(
                    fn_name,
                    format!(
                        "{fn_name}: expected a value after 'IgnoreCase'; provide true or false"
                    ),
                ));
            }
        }
        return parse_logical_value(fn_name, &rest[0]);
    }

    if !rest.len().is_multiple_of(2) {
        return Err(text_error(
            fn_name,
            format!(
                "{}: expected name-value pairs after the pattern argument (e.g., 'IgnoreCase', true)",
                fn_name
            ),
        ));
    }

    let mut ignore_case = None;
    for pair in rest.chunks_exact(2) {
        let name = value_to_owned_string(&pair[0], fn_name)?.ok_or_else(|| {
            text_error(
                fn_name,
                format!("{fn_name}: option names must be text scalars"),
            )
        })?;
        if !name.eq_ignore_ascii_case("ignorecase") {
            return Err(text_error(
                fn_name,
                format!("{fn_name}: unknown option '{name}'; supported option is 'IgnoreCase'"),
            ));
        }
        let value = parse_logical_value(fn_name, &pair[1])?;
        ignore_case = Some(value);
    }

    ignore_case.ok_or_else(|| {
        text_error(
            fn_name,
            format!("{fn_name}: expected 'IgnoreCase' option when providing name-value arguments"),
        )
    })
}

fn parse_logical_value(fn_name: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(text_error(
                    fn_name,
                    format!(
                        "{fn_name}: invalid numeric value for 'IgnoreCase'; expected a finite scalar"
                    ),
                ));
            }
            Ok(*n != 0.0)
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                return Err(text_error(
                    fn_name,
                    format!(
                        "{fn_name}: option values must be scalar logicals (received {} elements)",
                        array.data.len()
                    ),
                ));
            }
            Ok(array.data[0] != 0)
        }
        Value::Tensor(tensor) => {
            let len = tensor::tensor_element_len(tensor);
            if len != 1 {
                return Err(text_error(
                    fn_name,
                    format!(
                        "{fn_name}: option values must be scalar numeric values (received {} elements)",
                        len
                    ),
                ));
            }
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return Ok(!value.is_zero());
            }
            let value = tensor::tensor_value_f64(tensor, 0);
            if !value.is_finite() {
                return Err(text_error(
                    fn_name,
                    format!(
                        "{fn_name}: invalid numeric value for 'IgnoreCase'; expected a finite scalar"
                    ),
                ));
            }
            Ok(value != 0.0)
        }
        _ => {
            let text = value_to_owned_string(value, fn_name)?.ok_or_else(|| {
                text_error(
                    fn_name,
                    format!("{fn_name}: option values must be logical scalars"),
                )
            })?;
            match text.trim().to_ascii_lowercase().as_str() {
                "true" | "on" | "1" => Ok(true),
                "false" | "off" | "0" => Ok(false),
                other => Err(text_error(
                    fn_name,
                    format!(
                        "{fn_name}: invalid value '{other}' for 'IgnoreCase'; expected true or false"
                    ),
                )),
            }
        }
    }
}

pub(crate) fn value_to_owned_string(value: &Value, fn_name: &str) -> BuiltinResult<Option<String>> {
    match value {
        Value::CharArray(array) if array.shape.len() <= 2 && array.rows == 1 => array
            .try_row_string()
            .map_err(|source| text_decode_error(fn_name, source)),
        // Other scalar carriers cannot contain malformed UTF-16. This is a
        // shape/type probe; actual CharArray decoding must preserve its error.
        _ => Ok(String::try_from(value).ok()),
    }
}

fn text_decode_error(fn_name: &str, source: runmat_value::Utf16DecodeError) -> RuntimeError {
    build_runtime_error(format!("{fn_name}: malformed UTF-16 character input"))
        .with_builtin(fn_name)
        .with_source(source)
        .build()
}

fn text_error(fn_name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(fn_name).build()
}

pub(crate) fn logical_result(
    fn_name: &str,
    data: Vec<u8>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| text_error(fn_name, format!("{fn_name}: {e}")))
    }
}

fn make_text_element(text: StringElement) -> TextElement {
    match text {
        StringElement::Missing => TextElement::Missing,
        StringElement::Text(text) => TextElement::Text(text),
    }
}

fn cell_value_to_text(fn_name: &str, value: &Value) -> BuiltinResult<TextElement> {
    match value {
        Value::String(text) => Ok(make_text_element(text.clone())),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(make_text_element(array.data()[0].clone()))
        }
        Value::CharArray(array) if array.rows == 0 => Ok(TextElement::Text(String::new().into())),
        Value::CharArray(array) if array.rows == 1 => {
            let text = char_row_to_string(array, 0)
                .map_err(|source| text_decode_error(fn_name, source))?;
            Ok(TextElement::Text(text.into()))
        }
        _ => Err(text_error(
            fn_name,
            format!("{fn_name}: cell array elements must be character vectors or string scalars"),
        )),
    }
}
