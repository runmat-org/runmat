use super::ValueCodecError;
use runmat_builtins::{IntValue, IntegerStorage, NumericDType, SparseTensor, Tensor, Value};
use runmat_execution::value::{
    CallableValue, DenseValue, ElementType, ExceptionValue, InlineValue, SparseValue, StructField,
    ValuePayload,
};

pub fn encode_inline_value(value: &Value) -> Result<ValuePayload, ValueCodecError> {
    encode(value, "$")
}

fn encode(value: &Value, path: &str) -> Result<ValuePayload, ValueCodecError> {
    let inline = match value {
        Value::Int(value) => encode_integer_scalar(value),
        Value::Num(value) => InlineValue::F64Bits(value.to_bits()),
        Value::Complex(real, imaginary) => InlineValue::ComplexF64Bits {
            real: real.to_bits(),
            imaginary: imaginary.to_bits(),
        },
        Value::Bool(value) => InlineValue::Logical(*value),
        Value::LogicalArray(value) => InlineValue::Dense(DenseValue {
            element_type: ElementType::Logical,
            shape: shape(&value.shape, path)?,
            little_endian_data: value.data.clone(),
        }),
        Value::String(value) => InlineValue::String(value.clone()),
        Value::StringArray(value) => InlineValue::StringArray {
            shape: shape(&value.shape, path)?,
            values: value.data.clone(),
        },
        Value::CharArray(value) => InlineValue::Char {
            shape: vec![value.rows as u64, value.cols as u64],
            code_points: value.data.iter().map(|value| *value as u32).collect(),
        },
        Value::Tensor(value) => InlineValue::Dense(encode_tensor(value, path)?),
        Value::SparseTensor(value) => InlineValue::Sparse(encode_sparse(value, path)?),
        Value::ComplexTensor(value) => {
            let mut data = Vec::with_capacity(value.data.len().saturating_mul(16));
            for (real, imaginary) in &value.data {
                data.extend_from_slice(&real.to_bits().to_le_bytes());
                data.extend_from_slice(&imaginary.to_bits().to_le_bytes());
            }
            InlineValue::Dense(DenseValue {
                element_type: ElementType::ComplexF64,
                shape: shape(&value.shape, path)?,
                little_endian_data: data,
            })
        }
        Value::Symbolic(_) | Value::SymbolicArray(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "a registered symbolic codec is required",
            ))
        }
        Value::Cell(value) => InlineValue::Cell {
            shape: shape(&value.shape, path)?,
            values: value
                .data
                .iter()
                .enumerate()
                .map(|(index, value)| encode(value, &index_path(path, index)))
                .collect::<Result<_, _>>()?,
        },
        Value::Struct(value) => {
            let mut fields = value
                .fields
                .iter()
                .map(|(name, value)| {
                    Ok(StructField {
                        name: name.clone(),
                        value: encode(value, &field_path(path, name))?,
                    })
                })
                .collect::<Result<Vec<_>, ValueCodecError>>()?;
            fields.sort_by(|left, right| left.name.cmp(&right.name));
            InlineValue::Struct(fields)
        }
        Value::GpuTensor(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "GPU buffers require an object reference fenced to a worker and device",
            ))
        }
        Value::Object(_) | Value::ObjectArray(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "value classes require an explicitly registered immutable class codec",
            ))
        }
        Value::HandleObject(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "mutable handle identity cannot cross an execution boundary",
            ))
        }
        Value::Listener(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "event listeners are session-bound and cannot be transferred",
            ))
        }
        Value::OutputList(values) => InlineValue::OutputList(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| encode(value, &index_path(path, index)))
                .collect::<Result<_, _>>()?,
        ),
        Value::FunctionHandle(name) => callable("workspace", name, Vec::new()),
        Value::ExternalFunctionHandle(name) => callable("external", name, Vec::new()),
        Value::MethodFunctionHandle(name) => callable("method", name, Vec::new()),
        Value::BoundFunctionHandle { name, function } => {
            callable(&format!("bound:{function}"), name, Vec::new())
        }
        Value::Closure(closure) => {
            let captures = closure
                .captures
                .iter()
                .enumerate()
                .map(|(index, value)| encode(value, &format!("{path}.capture[{index}]")))
                .collect::<Result<_, _>>()?;
            let owner = closure
                .bound_function
                .map(|function| format!("bound:{function}"))
                .unwrap_or_else(|| "workspace".into());
            callable(&owner, &closure.function_name, captures)
        }
        Value::ClassRef(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "class metadata references are session-bound",
            ))
        }
        Value::MException(value) => InlineValue::Exception(ExceptionValue {
            identifier: value.identifier.clone(),
            message: value.message.clone(),
            stack: value.stack.clone(),
            causes: Vec::new(),
        }),
        Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => {
            return Err(ValueCodecError::unsupported(
                path,
                "execution handles are capabilities, not transferable values",
            ))
        }
    };
    Ok(ValuePayload::Inline(Box::new(inline)))
}

fn encode_integer_scalar(value: &IntValue) -> InlineValue {
    match value {
        IntValue::I8(value) => InlineValue::I8(*value),
        IntValue::I16(value) => InlineValue::I16(*value),
        IntValue::I32(value) => InlineValue::I32(*value),
        IntValue::I64(value) => InlineValue::I64(*value),
        IntValue::U8(value) => InlineValue::U8(*value),
        IntValue::U16(value) => InlineValue::U16(*value),
        IntValue::U32(value) => InlineValue::U32(*value),
        IntValue::U64(value) => InlineValue::U64(*value),
    }
}

fn encode_tensor(value: &Tensor, path: &str) -> Result<DenseValue, ValueCodecError> {
    let (element_type, little_endian_data) = if let Some(storage) = &value.integer_data {
        encode_integer_storage(storage)
    } else {
        match value.dtype {
            NumericDType::F64 => (
                ElementType::F64,
                value
                    .data
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect(),
            ),
            NumericDType::F32 => (
                ElementType::F32,
                value
                    .data
                    .iter()
                    .flat_map(|value| (*value as f32).to_bits().to_le_bytes())
                    .collect(),
            ),
            NumericDType::U8 => {
                if value.data.iter().any(|value| {
                    !value.is_finite() || *value < 0.0 || *value > 255.0 || value.fract() != 0.0
                }) {
                    return Err(ValueCodecError::invalid(
                        path,
                        "uint8 tensor compatibility data contains a non-uint8 value",
                    ));
                }
                (
                    ElementType::U8,
                    value.data.iter().map(|value| *value as u8).collect(),
                )
            }
            NumericDType::U16 | NumericDType::U32 => {
                return Err(ValueCodecError::invalid(
                    path,
                    "integer tensor is missing exact integer storage",
                ))
            }
        }
    };
    Ok(DenseValue {
        element_type,
        shape: shape(&value.shape, path)?,
        little_endian_data,
    })
}

fn encode_sparse(value: &SparseTensor, path: &str) -> Result<SparseValue, ValueCodecError> {
    let (element_type, little_endian_data) = match &value.integer_data {
        Some(storage) => encode_integer_storage(storage),
        None => (
            ElementType::F64,
            value
                .values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        ),
    };
    Ok(SparseValue {
        element_type,
        rows: value.rows as u64,
        columns: value.cols as u64,
        column_offsets: indices(&value.col_ptrs, path)?,
        row_indices: indices(&value.row_indices, path)?,
        little_endian_data,
    })
}

fn encode_integer_storage(storage: &IntegerStorage) -> (ElementType, Vec<u8>) {
    match storage {
        IntegerStorage::I8(values) => (ElementType::I8, values.iter().map(|v| *v as u8).collect()),
        IntegerStorage::I16(values) => (ElementType::I16, le_bytes(values, i16::to_le_bytes)),
        IntegerStorage::I32(values) => (ElementType::I32, le_bytes(values, i32::to_le_bytes)),
        IntegerStorage::I64(values) => (ElementType::I64, le_bytes(values, i64::to_le_bytes)),
        IntegerStorage::U8(values) => (ElementType::U8, values.clone()),
        IntegerStorage::U16(values) => (ElementType::U16, le_bytes(values, u16::to_le_bytes)),
        IntegerStorage::U32(values) => (ElementType::U32, le_bytes(values, u32::to_le_bytes)),
        IntegerStorage::U64(values) => (ElementType::U64, le_bytes(values, u64::to_le_bytes)),
    }
}

fn le_bytes<T: Copy, const N: usize>(values: &[T], encode: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|value| encode(*value)).collect()
}

fn callable(
    owner_identity: &str,
    qualified_name: &str,
    captures: Vec<ValuePayload>,
) -> InlineValue {
    InlineValue::Callable(CallableValue {
        owner_identity: owner_identity.into(),
        qualified_name: qualified_name.into(),
        callable_digest: CallableValue::identity_digest(owner_identity, qualified_name),
        captures,
    })
}

fn shape(shape: &[usize], path: &str) -> Result<Vec<u64>, ValueCodecError> {
    shape
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension)
                .map_err(|_| ValueCodecError::invalid(path, "shape exceeds portable limits"))
        })
        .collect()
}

fn indices(values: &[usize], path: &str) -> Result<Vec<u64>, ValueCodecError> {
    values
        .iter()
        .map(|value| {
            u64::try_from(*value)
                .map_err(|_| ValueCodecError::invalid(path, "index exceeds portable limits"))
        })
        .collect()
}

fn index_path(parent: &str, index: usize) -> String {
    format!("{parent}[{index}]")
}

fn field_path(parent: &str, field: &str) -> String {
    format!("{parent}.{field}")
}
