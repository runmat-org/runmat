use runmat_value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpectedScalarFieldName {
    Type,
    Encoding(runmat_value::Utf16DecodeError),
}

pub(crate) fn decode(value: &Value) -> Result<String, ExpectedScalarFieldName> {
    match value {
        Value::String(name) => name
            .as_str()
            .map(str::to_owned)
            .ok_or(ExpectedScalarFieldName::Type),
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data())
                .map_err(ExpectedScalarFieldName::Encoding)
        }
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .map(str::to_owned)
            .ok_or(ExpectedScalarFieldName::Type),
        _ => Err(ExpectedScalarFieldName::Type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CharArray, StringArray};

    #[test]
    fn decodes_each_scalar_text_representation_without_applying_name_policy() {
        assert_eq!(decode(&Value::from("field")).unwrap(), "field");
        assert_eq!(
            decode(&Value::CharArray(CharArray::new_row("field"))).unwrap(),
            "field"
        );
        let string = StringArray::new(vec!["field".into()], vec![1, 1]).unwrap();
        assert_eq!(decode(&Value::StringArray(string)).unwrap(), "field");
        assert_eq!(decode(&Value::from("")).unwrap(), "");
    }

    #[test]
    fn rejects_collection_and_nontext_representations() {
        let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        assert_eq!(
            decode(&Value::StringArray(strings)),
            Err(ExpectedScalarFieldName::Type)
        );
        assert_eq!(decode(&Value::Num(1.0)), Err(ExpectedScalarFieldName::Type));
    }
}
