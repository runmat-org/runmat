use super::{conversion_error, double_error_with_detail};

pub(super) fn convert(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Ok(Value::Num(value)),
        Value::Int(value) => Ok(Value::Num(value.to_f64())),
        Value::Bool(value) => Ok(Value::Num(if value { 1.0 } else { 0.0 })),
        Value::Tensor(value) => double_from_tensor(value),
        Value::SparseTensor(value) => double_from_sparse_tensor(value),
        Value::Complex(real, imag) => Ok(Value::Complex(real, imag)),
        Value::ComplexTensor(value) => double_from_complex_tensor(value),
        Value::LogicalArray(value) => double_from_logical(value),
        Value::CharArray(value) => double_from_char_array(value),
        Value::String(value) => Ok(Value::Num(parse_string_double(&value))),
        Value::StringArray(value) => double_from_string_array(value),
        Value::Symbolic(value) => value
            .numeric_constant_value()
            .map(Value::Num)
            .ok_or_else(|| conversion_error("sym")),
        Value::SymbolicArray(value) => double_from_symbolic_array(value),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) | Value::StructArray(_) => Err(conversion_error("struct")),
        Value::ObjectArray(value) => Err(conversion_error(value.class_name().display_name())),
        Value::Object(value) => Err(conversion_error(value.class_name.display_name())),
        Value::HandleObject(value) => Err(conversion_error(value.class_name.display_name())),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_) => Err(conversion_error("MException")),
        Value::Foreign(_) => Err(conversion_error("foreign")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
        Value::GpuTensor(_) => unreachable!("GPU conversion is routed before host dispatch"),
    }
}
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_builtins::DOUBLE_ERROR_INTERNAL;
use runmat_value::{
    CharArray, ComplexTensor, LogicalArray, NumericStorage, SparseTensor, StringArray,
    SymbolicArray, Tensor, Value,
};

pub(super) fn parse_string_double(text: &str) -> f64 {
    text.trim().parse::<f64>().unwrap_or(f64::NAN)
}

pub(super) fn double_from_string_array(array: StringArray) -> BuiltinResult<Value> {
    let shape = array.shape.clone();
    let data = array
        .data
        .iter()
        .map(|text| parse_string_double(text))
        .collect();
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

pub(super) fn double_from_logical(array: LogicalArray) -> BuiltinResult<Value> {
    let tensor = tensor::logical_to_tensor(&array)
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

pub(super) fn double_from_symbolic_array(array: SymbolicArray) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(array.data.len());
    for expr in array.data {
        data.push(
            expr.numeric_constant_value()
                .ok_or_else(|| conversion_error("sym"))?,
        );
    }
    Tensor::new(data, array.shape)
        .map(Value::Tensor)
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))
}

pub(super) fn double_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let values = tensor
        .into_numeric_storage()
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?
        .materialize_f64();
    Tensor::from_numeric_storage(NumericStorage::F64(values), shape)
        .map(Value::Tensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

pub(super) fn double_from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    ComplexTensor::new(tensor.materialize_f64(), shape)
        .map(Value::ComplexTensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

pub(super) fn double_from_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let values = sparse.materialize_f64();
    SparseTensor::new(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs,
        sparse.row_indices,
        values,
    )
    .map(Value::SparseTensor)
    .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))
}

pub(super) fn double_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

pub(super) fn double_from_gathered(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => double_from_tensor(tensor),
        Value::ComplexTensor(tensor) => double_from_complex_tensor(tensor),
        Value::LogicalArray(array) => double_from_logical(array),
        other => Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            format!("gather returned unsupported value {other:?}"),
        )),
    }
}
