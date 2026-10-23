use super::*;
use crate::execution::RootedValue;

pub(super) async fn datetime_indexing(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    payload: Value,
) -> BuiltinResult<RootedValue> {
    let Value::Object(object) = obj else {
        return Err(datetime_error(
            "datetime.subsref: receiver must be a datetime object",
        ));
    };
    let format = format_for_object(&object)?;
    let serials = serial_tensor_for_object(&object)?;

    let Value::Cell(cell) = payload else {
        return Err(datetime_error(
            "datetime.subsref: indexing payload must be a cell array",
        ));
    };
    if cell.data().is_empty() {
        return RootedValue::new(datetime_object_from_serial_tensor(
            runtime, serials, format,
        )?)
        .map_err(|error| error.into_runtime_error());
    }
    if cell.data().len() != 1 {
        return Err(datetime_error(
            "datetime.subsref: only linear datetime indexing is currently supported",
        ));
    }
    let selector = cell.data()[0].clone();
    let selector = match selector {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        Value::Int(value) => {
            Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?
        }
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        other => {
            return Err(datetime_error(format!(
                "datetime.subsref: unsupported index value {other:?}"
            )))
        }
    };
    let indexed = crate::perform_indexing(
        &Value::Tensor(serials),
        &tensor::tensor_values_f64(&selector),
    )
    .await
    .map_err(|err| datetime_error(format!("datetime.subsref: {}", err.message())))?;
    let indexed_serials = match indexed {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        Value::Tensor(tensor) => tensor,
        other => {
            return Err(datetime_error(format!(
                "datetime.subsref: unexpected indexing result {other:?}"
            )))
        }
    };
    RootedValue::new(datetime_object_from_serial_tensor(
        runtime,
        indexed_serials,
        format,
    )?)
    .map_err(|error| error.into_runtime_error())
}
