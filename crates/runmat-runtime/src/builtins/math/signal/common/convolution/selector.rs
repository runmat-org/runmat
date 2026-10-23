//! The shared text boundary for convolution shape selectors.

use runmat_value::Value;
use std::borrow::Cow;

pub(crate) enum ConvolutionSelectorText<'a> {
    Text(Cow<'a, str>),
    MalformedText,
    NotText,
}

pub(crate) fn convolution_selector_text(
    value: &Value,
) -> Result<ConvolutionSelectorText<'_>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(text) => ConvolutionSelectorText::Text(Cow::Borrowed(text.as_str())),
        Value::StringArray(array) if array.data().len() == 1 => {
            ConvolutionSelectorText::Text(Cow::Borrowed(array.data()[0].as_str()))
        }
        Value::CharArray(chars) if chars.rows == 1 && chars.shape.len() <= 2 => {
            ConvolutionSelectorText::Text(Cow::Owned(runmat_value::decode_utf16_code_units(
                chars.data(),
            )?))
        }
        Value::StringArray(_) | Value::CharArray(_) => ConvolutionSelectorText::MalformedText,
        _ => ConvolutionSelectorText::NotText,
    })
}
