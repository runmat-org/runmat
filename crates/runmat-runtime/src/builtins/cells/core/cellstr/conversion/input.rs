use runmat_value::{CellArray, CharArray, StringArray, SymbolicArray, SymbolicExpr, Value};

pub(super) enum Input {
    Character(CharArray),
    String(String),
    StringArray(StringArray),
    Cell(CellArray),
    Symbolic(SymbolicExpr),
    SymbolicArray(SymbolicArray),
    Unsupported,
}

pub(super) fn classify(value: Value) -> Input {
    match value {
        Value::CharArray(array) => Input::Character(array),
        Value::String(text) => Input::String(text),
        Value::StringArray(array) => Input::StringArray(array),
        Value::Cell(array) => Input::Cell(array),
        Value::Symbolic(expression) => Input::Symbolic(expression),
        Value::SymbolicArray(array) => Input::SymbolicArray(array),
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::GpuTensor(_)
        | Value::Object(_)
        | Value::ObjectArray(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::OutputList(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
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
