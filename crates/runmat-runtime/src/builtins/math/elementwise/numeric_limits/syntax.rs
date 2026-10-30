use runmat_value::Value;

pub(super) fn text(value: &Value, builtin: &'static str) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(characters)
            if characters.rows == 1
                && crate::builtins::common::shape::effective_rank(&characters.shape) <= 2 =>
        {
            Some(
                runmat_value::decode_utf16_code_units(characters.data()).map_err(|source| {
                    let mut error = super::errors::syntax(builtin, source.to_string());
                    error.source = Some(Box::new(source));
                    error
                })?,
            )
        }
        _ => None,
    })
}
