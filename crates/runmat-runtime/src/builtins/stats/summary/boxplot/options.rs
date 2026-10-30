//! Typed boxplot option admission and value decoding.
use super::admission::boxplot_matrix_dimensions;
use super::groups::{char_rows, number_label, numeric_scalar_label};
use super::*;
use crate::builtins::common::grouping_keys::format_integer;

#[cfg(test)]
impl BoxplotOptions {
    pub(super) fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        futures::executor::block_on(
            super::plan::BoxplotOptionPlan::admit(
                crate::execution::RootedValues::new(args)
                    .map_err(|error| error.into_runtime_error())?
                    .into(),
                None,
            )?
            .evaluate(),
        )
    }
}

pub(super) fn numeric_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(NAME, value.clone())
        .map_err(|err| invalid(format!("boxplot: {label} must be numeric ({err})")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

pub(super) fn scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => {
            let values = numeric_vector(other, label)?;
            if values.len() == 1 {
                Ok(values[0])
            } else {
                Err(invalid(format!("boxplot: {label} must be scalar")))
            }
        }
    }
}

pub(super) fn text_scalar(value: &Value, label: &str) -> BuiltinResult<String> {
    value_text_preserve_case(value, label).map(|text| text.to_ascii_lowercase())
}

pub(super) fn value_text_preserve_case(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("boxplot: {label} must be nonmissing text"))),
        Value::CharArray(array) if array.rows == 1 => {
            crate::builtins::stats::decode_utf16_with_error(array.data(), || {
                invalid(format!("boxplot: {label} is not valid UTF-16 text"))
            })
        }
        _ => Err(invalid(format!("boxplot: {label} must be text"))),
    }
}

pub(super) fn text_vector(value: &Value, label: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![text.as_str().to_owned()])
        }
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid(format!("boxplot: {label} must be nonmissing text")))
            })
            .collect(),
        Value::CharArray(array) => char_rows(array),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|value| value_text_preserve_case(value, label))
            .collect(),
        Value::Tensor(tensor) => (0..tensor::tensor_element_len(tensor))
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .map(numeric_scalar_label)
                    .ok_or_else(|| invalid(format!("boxplot: {label} storage is malformed")))
            })
            .collect(),
        Value::Num(n) => Ok(vec![number_label(n)]),
        Value::Int(i) => Ok(vec![format_integer(i)]),
        _ => Err(invalid(format!("boxplot: {label} must be a text vector"))),
    }
}

pub(super) fn parse_symbol(text: &str) -> BuiltinResult<(Option<MarkerStyle>, Option<Vec4>)> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return Ok((None, None));
    }
    let lower = trimmed.to_ascii_lowercase();
    let named = match lower.as_str() {
        "plus" => Some(MarkerStyle::Plus),
        "circle" | "o" => Some(MarkerStyle::Circle),
        "point" | "dot" => Some(MarkerStyle::Circle),
        "cross" | "x" => Some(MarkerStyle::Cross),
        "star" | "asterisk" => Some(MarkerStyle::Star),
        "square" => Some(MarkerStyle::Square),
        "diamond" => Some(MarkerStyle::Diamond),
        "triangle" | "uptriangle" | "downtriangle" | "righttriangle" | "lefttriangle" => {
            Some(MarkerStyle::Triangle)
        }
        "vline" | "vertical" => Some(MarkerStyle::Plus),
        "hline" | "horizontal" => Some(MarkerStyle::Square),
        _ => None,
    };
    if let Some(marker) = named {
        return Ok((Some(marker), None));
    }
    let mut marker = None;
    let mut color = None;
    for ch in trimmed.chars() {
        match ch {
            '+' => marker = Some(MarkerStyle::Plus),
            'o' | 'O' => marker = Some(MarkerStyle::Circle),
            '.' => marker = Some(MarkerStyle::Circle),
            'x' | 'X' => marker = Some(MarkerStyle::Cross),
            '*' => marker = Some(MarkerStyle::Star),
            's' | 'S' => marker = Some(MarkerStyle::Square),
            'd' | 'D' => marker = Some(MarkerStyle::Diamond),
            '^' | 'v' | '>' | '<' => marker = Some(MarkerStyle::Triangle),
            '|' => marker = Some(MarkerStyle::Plus),
            '_' => marker = Some(MarkerStyle::Square),
            'y' | 'm' | 'c' | 'r' | 'g' | 'b' | 'w' | 'k' => {
                color = Some(color_char(ch).expect("known color"))
            }
            ' ' => {}
            other => {
                return Err(invalid(format!(
                    "boxplot: unsupported Symbol character '{other}'"
                )));
            }
        }
    }
    Ok((marker, color))
}

pub(super) fn parse_colors(value: &Value) -> BuiltinResult<Vec<Vec4>> {
    if let Some(text) = match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned)
        }
        Value::CharArray(array) if array.rows == 1 => Some(
            crate::builtins::stats::decode_utf16_with_error(array.data(), || {
                invalid("boxplot: Colors must be valid UTF-16 text")
            })?,
        ),
        _ => None,
    } {
        let colors = text
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .map(|ch| {
                color_char(ch)
                    .ok_or_else(|| invalid(format!("boxplot: unsupported Colors character '{ch}'")))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        if colors.is_empty() {
            return Err(invalid("boxplot: Colors must not be empty"));
        }
        return Ok(colors);
    }
    let tensor = tensor::value_into_tensor_for(NAME, value.clone()).map_err(|err| {
        invalid(format!(
            "boxplot: Colors must be text or RGB matrix ({err})"
        ))
    })?;
    let (rows, cols) = boxplot_matrix_dimensions(&tensor);
    if cols != 3 || rows == 0 {
        return Err(invalid(
            "boxplot: Colors RGB matrix must have three columns",
        ));
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    let mut colors = Vec::with_capacity(rows);
    for row in 0..rows {
        let r = values[row];
        let g = values[rows + row];
        let b = values[2 * rows + row];
        if !(0.0..=1.0).contains(&r) || !(0.0..=1.0).contains(&g) || !(0.0..=1.0).contains(&b) {
            return Err(invalid("boxplot: RGB values must be in the range [0,1]"));
        }
        colors.push(Vec4::new(r as f32, g as f32, b as f32, 1.0));
    }
    Ok(colors)
}

fn color_char(ch: char) -> Option<Vec4> {
    match ch.to_ascii_lowercase() {
        'y' => Some(Vec4::new(1.0, 1.0, 0.0, 1.0)),
        'm' => Some(Vec4::new(1.0, 0.0, 1.0, 1.0)),
        'c' => Some(Vec4::new(0.0, 1.0, 1.0, 1.0)),
        'r' => Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        'g' => Some(Vec4::new(0.0, 0.5, 0.0, 1.0)),
        'b' => Some(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        'w' => Some(Vec4::new(1.0, 1.0, 1.0, 1.0)),
        'k' => Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        _ => None,
    }
}
