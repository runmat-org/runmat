use runmat_value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextScalarError {
    NotTextScalar,
    Encoding(runmat_value::Utf16DecodeError),
}

impl TextScalarError {
    pub(crate) fn attach(self, mut error: crate::RuntimeError) -> crate::RuntimeError {
        if let Self::Encoding(source) = self {
            error.source = Some(Box::new(source));
        }
        error
    }
}

pub(crate) fn text_scalar(value: &Value) -> Result<String, TextScalarError> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            runmat_value::decode_utf16_code_units(array.data()).map_err(TextScalarError::Encoding)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        _ => Err(TextScalarError::NotTextScalar),
    }
}
