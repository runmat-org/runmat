use super::invalid_argument;
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_builtins::RidgeScaling;
use runmat_value::{NumericScalar, Tensor, Value};

pub(super) async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value).await?;
    tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid_argument(format!("ridge: {err}")))
}

pub(super) async fn parse_scaled(value: &Value) -> BuiltinResult<RidgeScaling> {
    let gathered = gather_if_needed_async(value).await?;
    let parsed = match gathered {
        Value::Bool(value) => Some(RidgeScaling::from_bool(value)),
        Value::Num(value) => RidgeScaling::from_float(value),
        Value::Int(value) => RidgeScaling::from_integer(value.to_i128()),
        Value::Tensor(tensor) => {
            if tensor.host_buffer().len() != 1 {
                return Err(invalid_argument(format!(
                    "ridge: scaled must be 0, 1, false, or true, got {:?}",
                    Value::Tensor(tensor)
                )));
            }
            let value = tensor
                .host_buffer()
                .value_at(0)
                .ok_or_else(|| invalid_argument("ridge: scaled must contain one value"))?;
            match value {
                NumericScalar::F64(value) => RidgeScaling::from_float(value),
                NumericScalar::F32(value) => RidgeScaling::from_float(f64::from(value)),
                integer => integer
                    .into_int_value()
                    .and_then(|value| RidgeScaling::from_integer(value.to_i128())),
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "ridge: scaled must be 0, 1, false, or true, got {other:?}"
            )));
        }
    };
    parsed.ok_or_else(|| invalid_argument("ridge: scaled must be 0, 1, false, or true"))
}
