use super::*;
pub(super) trait TypeName {
    fn type_name(&self) -> &'static str;
}

impl TypeName for Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Num(_) => "double",
            Value::Complex(_, _) => "complex double",
            Value::Bool(_) => "logical",
            Value::LogicalArray(_) => "logical array",
            Value::String(_) => "string",
            Value::StringArray(_) => "string array",
            Value::CharArray(_) => "char array",
            Value::Symbolic(_) => "sym",
            Value::SymbolicArray(_) => "symbolic array",
            Value::Tensor(tensor) => numeric_array_type_name(tensor.numeric_dtype(), ""),
            Value::SparseTensor(tensor) if tensor.is_logical() => "sparse logical array",
            Value::SparseTensor(tensor) => numeric_array_type_name(
                tensor
                    .numeric_dtype()
                    .expect("non-logical sparse tensor has a numeric dtype"),
                "sparse",
            ),
            Value::ComplexTensor(tensor) => {
                numeric_array_type_name(tensor.numeric_dtype(), "complex")
            }
            Value::Cell(_) => "cell array",
            Value::Struct(_) | Value::StructArray(_) => "struct",
            Value::GpuTensor(_) => "gpuArray",
            Value::ObjectArray(_) | Value::Object(_) => "object",
            Value::HandleObject(_) => "handle object",
            Value::Listener(_) => "listener",
            Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::ScopedFunctionHandle(_) => "function handle",
            Value::Closure(_) => "closure",
            Value::ClassRef(_) => "class reference",
            Value::MException(_) => "MException",
            Value::OutputList(_) => "output list",
            Value::Future(_) => "future",
            Value::Task(_) => "task",
            Value::Pool(_) => "pool",
            Value::Job(_) => "job",
            Value::Distributed(_) => "distributed",
            Value::Composite(_) => "Composite",
            Value::Foreign(_) => "foreign",
        }
    }
}

pub(super) fn numeric_array_type_name(dtype: NumericDType, kind: &str) -> &'static str {
    match (kind, dtype) {
        ("", NumericDType::F64) => "double array",
        ("", NumericDType::F32) => "single array",
        ("", NumericDType::I8) => "int8 array",
        ("", NumericDType::I16) => "int16 array",
        ("", NumericDType::I32) => "int32 array",
        ("", NumericDType::I64) => "int64 array",
        ("", NumericDType::U8) => "uint8 array",
        ("", NumericDType::U16) => "uint16 array",
        ("", NumericDType::U32) => "uint32 array",
        ("", NumericDType::U64) => "uint64 array",
        ("sparse", NumericDType::F64) => "sparse double array",
        ("sparse", NumericDType::F32) => "sparse single array",
        ("sparse", NumericDType::I8) => "sparse int8 array",
        ("sparse", NumericDType::I16) => "sparse int16 array",
        ("sparse", NumericDType::I32) => "sparse int32 array",
        ("sparse", NumericDType::I64) => "sparse int64 array",
        ("sparse", NumericDType::U8) => "sparse uint8 array",
        ("sparse", NumericDType::U16) => "sparse uint16 array",
        ("sparse", NumericDType::U32) => "sparse uint32 array",
        ("sparse", NumericDType::U64) => "sparse uint64 array",
        ("complex", NumericDType::F64) => "complex double array",
        ("complex", NumericDType::F32) => "complex single array",
        ("complex", NumericDType::I8) => "complex int8 array",
        ("complex", NumericDType::I16) => "complex int16 array",
        ("complex", NumericDType::I32) => "complex int32 array",
        ("complex", NumericDType::I64) => "complex int64 array",
        ("complex", NumericDType::U8) => "complex uint8 array",
        ("complex", NumericDType::U16) => "complex uint16 array",
        ("complex", NumericDType::U32) => "complex uint32 array",
        ("complex", NumericDType::U64) => "complex uint64 array",
        _ => "numeric array",
    }
}
