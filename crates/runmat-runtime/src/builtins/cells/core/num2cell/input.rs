use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, ObjectArray, SparseTensor, StringArray,
    SymbolicArray, Tensor, Value,
};

pub(super) async fn gather_top_level(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => crate::gather_if_needed_async(&Value::GpuTensor(handle)).await,
        value => Ok(value),
    }
}

pub(super) enum Input {
    Numeric(Tensor),
    Sparse(SparseTensor),
    Complex(ComplexTensor),
    Logical(LogicalArray),
    String(StringArray),
    Character(CharArray),
    Symbolic(SymbolicArray),
    Cell(CellArray),
    Object(ObjectArray),
    Scalar(Value),
    Unsupported,
}

pub(super) fn classify(value: Value) -> Input {
    match value {
        Value::Tensor(value) => Input::Numeric(value),
        Value::SparseTensor(value) => Input::Sparse(value),
        Value::ComplexTensor(value) => Input::Complex(value),
        Value::LogicalArray(value) => Input::Logical(value),
        Value::StringArray(value) => Input::String(value),
        Value::CharArray(value) => Input::Character(value),
        Value::SymbolicArray(value) => Input::Symbolic(value),
        Value::Cell(value) => Input::Cell(value),
        Value::ObjectArray(value) => Input::Object(value),
        value @ (Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::String(_)
        | Value::Symbolic(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)) => Input::Scalar(value),
        Value::GpuTensor(_)
        | Value::Listener(_)
        | Value::OutputList(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_) => Input::Unsupported,
    }
}
