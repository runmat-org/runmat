use runmat_builtins::IntValue;

pub(crate) fn int_value_to_json(value: &IntValue) -> serde_json::Value {
    let number = match value {
        IntValue::I8(value) => serde_json::Number::from(*value),
        IntValue::I16(value) => serde_json::Number::from(*value),
        IntValue::I32(value) => serde_json::Number::from(*value),
        IntValue::I64(value) => serde_json::Number::from(*value),
        IntValue::U8(value) => serde_json::Number::from(*value),
        IntValue::U16(value) => serde_json::Number::from(*value),
        IntValue::U32(value) => serde_json::Number::from(*value),
        IntValue::U64(value) => serde_json::Number::from(*value),
    };
    serde_json::Value::Number(number)
}
