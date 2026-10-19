use super::*;

pub(super) enum TopKInput {
    Real(Tensor),
    Complex(ComplexTensor),
    Logical(LogicalArray),
}

pub(super) async fn gather_topk_input(kind: TopKKind, value: Value) -> BuiltinResult<TopKInput> {
    let host = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(|err| topk_internal(kind, err.message()))?;
    match host {
        Value::Tensor(tensor) => Ok(TopKInput::Real(tensor)),
        Value::LogicalArray(logical) => Ok(TopKInput::Logical(logical)),
        Value::Num(value) => Ok(TopKInput::Real(
            Tensor::new(vec![value], vec![1, 1]).map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Int(value) => Ok(TopKInput::Real(
            Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Bool(value) => Ok(TopKInput::Logical(
            LogicalArray::new(vec![u8::from(value)], vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::Complex(re, im) => Ok(TopKInput::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|message| topk_internal(kind, message))?,
        )),
        Value::ComplexTensor(tensor) => Ok(TopKInput::Complex(tensor)),
        _ => Err(topk_invalid_input(
            kind,
            "expected numeric, logical, or complex input",
        )),
    }
}
