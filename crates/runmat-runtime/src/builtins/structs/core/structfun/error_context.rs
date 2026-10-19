use runmat_value::{StructValue, Value};

use super::callback::CallbackFailure;

pub(super) fn value(error: &CallbackFailure, index: usize, field: &str) -> Value {
    let mut context = StructValue::new();
    context.insert(
        "identifier",
        Value::String(error.identifier().unwrap_or("").into()),
    );
    context.insert("message", Value::String(error.message().into_owned()));
    context.insert("index", Value::Num((index + 1) as f64));
    context.insert("field", Value::String(field.into()));
    Value::Struct(context)
}
