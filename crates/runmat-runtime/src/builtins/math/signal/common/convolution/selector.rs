//! The shared text boundary for convolution shape selectors.

use runmat_value::Value;

pub(crate) enum ConvolutionSelectorText {
    Text(String),
    MalformedText,
    NotText,
}

pub(crate) fn convolution_selector_text(value: &Value) -> ConvolutionSelectorText {
    match value {
        Value::String(text) => ConvolutionSelectorText::Text(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => {
            ConvolutionSelectorText::Text(array.data[0].clone())
        }
        Value::CharArray(chars) if chars.rows == 1 && chars.shape.len() <= 2 => {
            ConvolutionSelectorText::Text(chars.data.iter().collect())
        }
        Value::StringArray(_) | Value::CharArray(_) => ConvolutionSelectorText::MalformedText,
        _ => ConvolutionSelectorText::NotText,
    }
}
