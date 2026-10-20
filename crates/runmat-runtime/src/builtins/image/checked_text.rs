//! Scalar text admitted by image I/O before file or network effects.

use runmat_value::Value;

pub(super) fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Some(array.data()[0].as_str().to_owned())
        }
        Value::CharArray(array) if array.rows == 1 => Some(array.data().iter().collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_numeric_and_logical_filename_coercions() {
        assert_eq!(scalar_text(&Value::Num(42.0)), None);
        assert_eq!(scalar_text(&Value::Bool(true)), None);
        assert_eq!(
            scalar_text(&Value::String("part.png".into())),
            Some("part.png".into())
        );
    }
}
