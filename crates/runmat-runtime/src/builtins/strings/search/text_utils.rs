//! Shared helpers for string search builtins (contains, startsWith, etc.).

use std::convert::TryFrom;

use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray, Value};

use crate::builtins::common::tensor;
use crate::builtins::strings::common::{char_row_to_string, is_missing_string};

#[derive(Clone)]
pub(crate) struct TextCollection {
    pub(crate) elements: Vec<TextElement>,
    pub(crate) shape: Vec<usize>,
    pub(crate) is_cell: bool,
}

#[derive(Clone)]
pub(crate) enum TextElement {
    Missing,
    Text(String),
}

impl TextCollection {
    pub(crate) fn from_subject(fn_name: &str, value: Value) -> Result<Self, String> {
        Self::from_value(fn_name, value, true)
    }

    pub(crate) fn from_pattern(fn_name: &str, value: Value) -> Result<Self, String> {
        Self::from_value(fn_name, value, false)
    }

    fn from_value(fn_name: &str, value: Value, is_subject: bool) -> Result<Self, String> {
        let collection = match value {
            Value::StringArray(array) => Ok(Self::from_string_array(array)),
            Value::String(text) => Ok(Self::from_string_scalar(text)),
            Value::CharArray(array) => Ok(Self::from_char_array(array)),
            Value::Cell(cell) => Self::from_cell_array(fn_name, cell),
            _ => {
                let descriptor = if is_subject {
                    "first argument"
                } else {
                    "pattern"
                };
                Err(format!("{fn_name}: {descriptor} must be text (string array, character array, or cell array of character vectors)"))
            }
        }?;

        if collection.elements.is_empty()
            || tensor::element_count(&collection.shape) == collection.elements.len()
        {
            Ok(collection)
        } else if is_subject {
            Err(format!(
                "{fn_name}: first argument must be text (string array, character array, or cell array of character vectors)"
            ))
        } else {
            Err(format!(
                "{fn_name}: pattern must be text (string array, character array, or cell array of character vectors)"
            ))
        }
    }

    fn from_string_array(array: StringArray) -> Self {
        let StringArray { data, shape, .. } = array;
        let elements = data.into_iter().map(make_text_element).collect::<Vec<_>>();
        Self {
            elements,
            shape,
            is_cell: false,
        }
    }

    fn from_string_scalar(text: String) -> Self {
        Self {
            elements: vec![make_text_element(text)],
            shape: vec![1, 1],
            is_cell: false,
        }
    }

    fn from_char_array(array: CharArray) -> Self {
        if array.rows == 0 {
            return Self {
                elements: Vec::new(),
                shape: vec![0, 1],
                is_cell: false,
            };
        }
        if array.rows == 1 {
            let text = char_row_to_string(&array, 0);
            return Self {
                elements: vec![TextElement::Text(text)],
                shape: vec![1, 1],
                is_cell: false,
            };
        }
        let mut elements = Vec::with_capacity(array.rows);
        for row in 0..array.rows {
            elements.push(TextElement::Text(char_row_to_string(&array, row)));
        }
        Self {
            elements,
            shape: vec![array.rows, 1],
            is_cell: false,
        }
    }

    fn from_cell_array(fn_name: &str, cell: CellArray) -> Result<Self, String> {
        let CellArray {
            data, rows, cols, ..
        } = cell;
        let mut elements = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                let idx = row * cols + col;
                let value = &*data[idx];
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

pub(crate) fn parse_ignore_case(fn_name: &str, rest: &[Value]) -> Result<bool, String> {
    if rest.is_empty() {
        return Ok(false);
    }

    if rest.len() == 1 {
        if let Some(name) = value_to_owned_string(&rest[0]) {
            if name.eq_ignore_ascii_case("ignorecase") {
                return Err(format!(
                    "{fn_name}: expected a value after 'IgnoreCase'; provide true or false"
                ));
            }
        }
        return parse_logical_value(fn_name, &rest[0]);
    }

    if rest.len() % 2 != 0 {
        return Err(format!(
            "{}: expected name-value pairs after the pattern argument (e.g., 'IgnoreCase', true)",
            fn_name
        ));
    }

    let mut ignore_case = None;
    for pair in rest.chunks(2) {
        let name = value_to_owned_string(&pair[0])
            .ok_or_else(|| format!("{fn_name}: option names must be text scalars"))?;
        if !name.eq_ignore_ascii_case("ignorecase") {
            return Err(format!(
                "{fn_name}: unknown option '{name}'; supported option is 'IgnoreCase'"
            ));
        }
        let value = parse_logical_value(fn_name, &pair[1])?;
        ignore_case = Some(value);
    }

    ignore_case.ok_or_else(|| {
        format!("{fn_name}: expected 'IgnoreCase' option when providing name-value arguments")
    })
}

fn parse_logical_value(fn_name: &str, value: &Value) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!(
                    "{fn_name}: invalid numeric value for 'IgnoreCase'; expected a finite scalar"
                ));
            }
            Ok(*n != 0.0)
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                return Err(format!(
                    "{fn_name}: option values must be scalar logicals (received {} elements)",
                    array.data.len()
                ));
            }
            Ok(array.data[0] != 0)
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(format!(
                    "{fn_name}: option values must be scalar numeric values (received {} elements)",
                    tensor.data.len()
                ));
            }
            let value = tensor.data[0];
            if !value.is_finite() {
                return Err(format!(
                    "{fn_name}: invalid numeric value for 'IgnoreCase'; expected a finite scalar"
                ));
            }
            Ok(value != 0.0)
        }
        _ => {
            let text = value_to_owned_string(value)
                .ok_or_else(|| format!("{fn_name}: option values must be logical scalars"))?;
            match text.trim().to_ascii_lowercase().as_str() {
                "true" | "on" | "1" => Ok(true),
                "false" | "off" | "0" => Ok(false),
                other => Err(format!(
                    "{fn_name}: invalid value '{other}' for 'IgnoreCase'; expected true or false"
                )),
            }
        }
    }
}

pub(crate) fn value_to_owned_string(value: &Value) -> Option<String> {
    String::try_from(value).ok()
}

pub(crate) fn logical_result(
    fn_name: &str,
    data: Vec<u8>,
    shape: Vec<usize>,
) -> Result<Value, String> {
    if data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| format!("{fn_name}: {e}"))
    }
}

fn make_text_element(text: String) -> TextElement {
    if is_missing_string(&text) {
        TextElement::Missing
    } else {
        TextElement::Text(text)
    }
}

fn cell_value_to_text(fn_name: &str, value: &Value) -> Result<TextElement, String> {
    match value {
        Value::String(text) => Ok(make_text_element(text.clone())),
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok(make_text_element(array.data[0].clone()))
        }
        Value::CharArray(array) if array.rows == 0 => Ok(TextElement::Text(String::new())),
        Value::CharArray(array) if array.rows == 1 => {
            Ok(TextElement::Text(char_row_to_string(array, 0)))
        }
        _ => Err(format!(
            "{fn_name}: cell array elements must be character vectors or string scalars"
        )),
    }
}
