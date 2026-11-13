use super::{root_value, RootedValue, Value};

/// A MAT entry is either already a rooted value or still needs the object
/// restoration future (including its loadobj dispatch and cancellation fence).
pub(crate) enum MatLoadRestoration {
    Ready(MatLoadLeaf),
    Recursive(Value),
}

pub(crate) struct MatLoadLeaf(Value);

impl MatLoadLeaf {
    pub(crate) fn into_rooted(self) -> crate::BuiltinResult<RootedValue> {
        root_value(self.0)
    }
}

pub(crate) fn classify_mat_load_restoration(value: Value) -> MatLoadRestoration {
    match &value {
        // Aggregate values retain recursive depth and child-root checks.
        Value::Cell(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::Object(_)
        | Value::ObjectArray(_)
        | Value::OutputList(_)
        | Value::Closure(_)
        | Value::MException(_) => MatLoadRestoration::Recursive(value),
        // Closed, flat value classes: the recursive restorer's terminal branch
        // only roots these. Rooting synchronously avoids its large poll frame.
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::GpuTensor(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::ScopedFunctionHandle(_)
        | Value::ClassRef(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_) => MatLoadRestoration::Ready(MatLoadLeaf(value)),
    }
}

#[cfg(test)]
mod tests;
