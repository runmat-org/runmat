use super::storage::{
    char_array_to_tensor, numeric_scalar_to_f32, single_complex_tensor_to_host,
    single_tensor_to_host,
};
use super::{conversion_error, single_error_with_detail};

pub(super) fn convert(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => single_scalar_value(value as f32),
        Value::Int(value) => single_scalar_value(super::storage::int_value_to_f32(&value)),
        Value::Bool(value) => single_scalar_value(if value { 1.0 } else { 0.0 }),
        Value::Tensor(value) => single_from_tensor(value),
        Value::SparseTensor(value) => single_from_sparse_tensor(value),
        Value::Complex(real, imag) => single_complex_scalar_value(real as f32, imag as f32),
        Value::ComplexTensor(value) => single_from_complex_tensor(value),
        Value::LogicalArray(value) => single_from_logical_array(value),
        Value::CharArray(value) => single_from_char_array(value),
        Value::String(_) | Value::StringArray(_) => Err(conversion_error("string")),
        Value::Symbolic(value) => value
            .numeric_constant_value()
            .map(|value| single_scalar_value(value as f32))
            .transpose()?
            .ok_or_else(|| conversion_error("sym")),
        Value::SymbolicArray(value) => single_from_symbolic_array(value),
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
use runmat_builtins::SINGLE_ERROR_INTERNAL;
use runmat_value::{
    CharArray, ComplexTensor, LogicalArray, NumericStorage, SparseTensor, SymbolicArray, Tensor,
    Value,
};

pub(super) fn single_scalar_value(value: f32) -> BuiltinResult<Value> {
    Tensor::from_f32(vec![value], vec![1, 1])
        .map(Value::Tensor)
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))
}

pub(super) fn single_complex_scalar_value(real: f32, imag: f32) -> BuiltinResult<Value> {
    ComplexTensor::from_f32(vec![(real, imag)], vec![1, 1])
        .map(Value::ComplexTensor)
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))
}

pub(super) fn single_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    single_tensor_to_host(tensor).map(Value::Tensor)
}

pub(super) fn single_from_symbolic_array(array: SymbolicArray) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(array.data.len());
    for expr in array.data {
        let value = expr
            .numeric_constant_value()
            .ok_or_else(|| conversion_error("sym"))?;
        data.push(value as f32);
    }
    Tensor::from_numeric_storage(NumericStorage::F32(data), array.shape)
        .map(Value::Tensor)
        .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))
}

pub(super) fn single_from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    single_complex_tensor_to_host(tensor).map(Value::ComplexTensor)
}

pub(super) fn single_from_sparse_tensor(sparse: SparseTensor) -> BuiltinResult<Value> {
    let values = (0..sparse.nnz())
        .map(|index| {
            sparse
                .numeric_value_at(index)
                .map(numeric_scalar_to_f32)
                .ok_or_else(|| {
                    single_error_with_detail(
                        &SINGLE_ERROR_INTERNAL,
                        "sparse value storage is inconsistent",
                    )
                })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    SparseTensor::new_f32(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs,
        sparse.row_indices,
        values,
    )
    .map(Value::SparseTensor)
    .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))
}

pub(super) fn single_from_logical_array(array: LogicalArray) -> BuiltinResult<Value> {
    let tensor = tensor::logical_to_tensor(&array)
        .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
    single_tensor_to_host(tensor).map(Value::Tensor)
}

pub(super) fn single_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let tensor = char_array_to_tensor(&chars)?;
    single_tensor_to_host(tensor).map(Value::Tensor)
}

pub(super) fn single_from_gathered(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => single_from_tensor(tensor),
        Value::ComplexTensor(tensor) => single_from_complex_tensor(tensor),
        Value::LogicalArray(array) => single_from_logical_array(array),
        other => Err(single_error_with_detail(
            &SINGLE_ERROR_INTERNAL,
            format!("gather returned unsupported value {other:?}"),
        )),
    }
}
