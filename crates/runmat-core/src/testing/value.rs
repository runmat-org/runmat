use runmat_value::{CellArray, StructValue, Tensor, Value};

pub(super) fn from_json(value: &serde_json::Value) -> Result<Value, String> {
    match value {
        serde_json::Value::Null => Tensor::new(Vec::new(), vec![0, 0]).map(Value::Tensor),
        serde_json::Value::Bool(value) => Ok(Value::Bool(*value)),
        serde_json::Value::Number(value) => value
            .as_f64()
            .map(Value::Num)
            .ok_or_else(|| "test parameter number is outside f64 range".into()),
        serde_json::Value::String(value) => Ok(Value::String(value.clone())),
        serde_json::Value::Array(values) => {
            if values.iter().all(serde_json::Value::is_number) {
                let data = values
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .ok_or_else(|| "test parameter number is outside f64 range".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let len = data.len();
                Tensor::new(data, vec![1, len]).map(Value::Tensor)
            } else {
                let data = values
                    .iter()
                    .map(from_json)
                    .collect::<Result<Vec<_>, _>>()?;
                let len = data.len();
                CellArray::new(data, 1, len).map(Value::Cell)
            }
        }
        serde_json::Value::Object(values) => {
            let mut structure = StructValue::new();
            for (name, value) in values {
                structure.fields.insert(name.clone(), from_json(value)?);
            }
            Ok(Value::Struct(structure))
        }
    }
}
