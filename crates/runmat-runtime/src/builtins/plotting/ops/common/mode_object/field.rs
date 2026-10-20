use super::{Field, FieldValue};
use runmat_value::Value;

impl Field<'_> {
    pub(crate) fn value(&self) -> &Value {
        match &self.value {
            FieldValue::Value(Some(value)) => value,
            _ => unreachable!("mode-object field was not prepared"),
        }
    }
}
