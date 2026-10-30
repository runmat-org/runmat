//! Scalar text admitted by image I/O before file or network effects.

use runmat_value::Value;

pub(super) fn scalar_text(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
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
        Value::CharArray(array) if array.rows == 1 => {
            Some(runmat_value::decode_utf16_code_units(array.data())?)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_numeric_and_logical_filename_coercions() {
        assert_eq!(scalar_text(&Value::Num(42.0)).unwrap(), None);
        assert_eq!(scalar_text(&Value::Bool(true)).unwrap(), None);
        assert_eq!(
            scalar_text(&Value::String("part.png".into())).unwrap(),
            Some("part.png".into())
        );
        let supplementary = runmat_value::CharArray::new_row("🦀.png");
        assert_eq!(
            scalar_text(&Value::CharArray(supplementary))
                .unwrap()
                .as_deref(),
            Some("🦀.png")
        );
        let malformed = runmat_value::CharArray::from_code_units(
            vec![runmat_value::Utf16CodeUnit::from_raw(0xd800)],
            1,
            1,
        )
        .unwrap();
        assert!(scalar_text(&Value::CharArray(malformed)).is_err());
    }
}
