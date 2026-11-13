use super::*;

pub(super) fn tensor_value(data: Vec<f64>, shape: Vec<usize>, label: &str) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| internal(format!("regress: failed to construct {label}: {err}")))
}
