use runmat_execution::value::{
    CallableValue, DenseValue, ElementType, ExceptionValue, InlineValue, SparseValue, ValueLimits,
    ValuePayload,
};
use runmat_value::{
    CellArray, CharArray, Closure, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    LogicalArray, MException, SparseTensor, StringArray, StructValue, Tensor, Value,
};

use super::ValueCodecError;

pub fn decode_inline_value(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    payload
        .validate(ValueLimits::default())
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    decode(payload, "$")
}

fn decode(payload: &ValuePayload, path: &str) -> Result<Value, ValueCodecError> {
    let ValuePayload::Inline(value) = payload else {
        return Err(ValueCodecError::unsupported(
            path,
            "object references require an execution object-store decoder",
        ));
    };
    match value.as_ref() {
        InlineValue::Null => Err(ValueCodecError::unsupported(
            path,
            "MATLAB has no null runtime value",
        )),
        InlineValue::Logical(value) => Ok(Value::Bool(*value)),
        InlineValue::F64Bits(value) => Ok(Value::Num(f64::from_bits(*value))),
        InlineValue::I8(value) => Ok(Value::Int(IntValue::I8(*value))),
        InlineValue::I16(value) => Ok(Value::Int(IntValue::I16(*value))),
        InlineValue::I32(value) => Ok(Value::Int(IntValue::I32(*value))),
        InlineValue::I64(value) => Ok(Value::Int(IntValue::I64(*value))),
        InlineValue::U8(value) => Ok(Value::Int(IntValue::U8(*value))),
        InlineValue::U16(value) => Ok(Value::Int(IntValue::U16(*value))),
        InlineValue::U32(value) => Ok(Value::Int(IntValue::U32(*value))),
        InlineValue::U64(value) => Ok(Value::Int(IntValue::U64(*value))),
        InlineValue::ComplexF64Bits { real, imaginary } => Ok(Value::Complex(
            f64::from_bits(*real),
            f64::from_bits(*imaginary),
        )),
        InlineValue::String(value) => Ok(Value::String(value.clone())),
        InlineValue::StringArray { shape, values } => Ok(Value::StringArray(
            StringArray::new(values.clone(), usize_shape(shape, path)?)
                .map_err(|error| ValueCodecError::invalid(path, error))?,
        )),
        InlineValue::Char { shape, code_points } => {
            if shape.len() != 2 {
                return Err(ValueCodecError::invalid(
                    path,
                    "char arrays require exactly two dimensions",
                ));
            }
            let shape = usize_shape(shape, path)?;
            let chars = code_points
                .iter()
                .map(|value| {
                    char::from_u32(*value)
                        .ok_or_else(|| ValueCodecError::invalid(path, "invalid Unicode scalar"))
                })
                .collect::<Result<_, _>>()?;
            Ok(Value::CharArray(
                CharArray::new(chars, shape[0], shape[1])
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
            ))
        }
        InlineValue::Dense(value) => decode_dense(value, path),
        InlineValue::Sparse(value) => decode_sparse(value, path),
        InlineValue::Symbolic(_) => Err(ValueCodecError::unsupported(
            path,
            "the registered symbolic codec is unavailable in this runtime",
        )),
        InlineValue::Cell { shape, values } => Ok(Value::Cell(
            CellArray::new_with_shape(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| decode(value, &format!("{path}[{index}]")))
                    .collect::<Result<_, _>>()?,
                usize_shape(shape, path)?,
            )
            .map_err(|error| ValueCodecError::invalid(path, error))?,
        )),
        InlineValue::Struct(fields) => {
            let mut value = StructValue::new();
            for field in fields {
                value.insert(
                    field.name.clone(),
                    decode(&field.value, &format!("{path}.{}", field.name))?,
                );
            }
            Ok(Value::Struct(value))
        }
        InlineValue::OutputList(values) => Ok(Value::OutputList(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| decode(value, &format!("{path}[{index}]")))
                .collect::<Result<_, _>>()?,
        )),
        InlineValue::Exception(value) => decode_exception(value, path),
        InlineValue::Callable(value) => decode_callable(value, path),
        InlineValue::ImmutableValueClass(_) => Err(ValueCodecError::unsupported(
            path,
            "the registered immutable value-class codec is unavailable in this runtime",
        )),
    }
}

fn decode_dense(value: &DenseValue, path: &str) -> Result<Value, ValueCodecError> {
    let shape = usize_shape(&value.shape, path)?;
    match value.element_type {
        ElementType::Logical => Ok(Value::LogicalArray(
            LogicalArray::new(value.little_endian_data.clone(), shape)
                .map_err(|error| ValueCodecError::invalid(path, error))?,
        )),
        ElementType::F32 => Ok(Value::Tensor(
            Tensor::from_f32(decode_f32(&value.little_endian_data, path)?, shape)
                .map_err(|error| ValueCodecError::invalid(path, error))?,
        )),
        ElementType::F64 => Ok(Value::Tensor(
            Tensor::new(decode_f64(&value.little_endian_data, path)?, shape)
                .map_err(|error| ValueCodecError::invalid(path, error))?,
        )),
        ElementType::ComplexF64 => {
            let words = decode_words::<8>(&value.little_endian_data, path)?;
            let data = words
                .chunks_exact(2)
                .map(|pair| (f64::from_bits(pair[0]), f64::from_bits(pair[1])))
                .collect();
            Ok(Value::ComplexTensor(
                ComplexTensor::new(data, shape)
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
            ))
        }
        ElementType::ComplexF32 => {
            let words = decode_arrays::<4>(&value.little_endian_data, path)?;
            let data = words
                .chunks_exact(2)
                .map(|pair| {
                    (
                        f32::from_bits(u32::from_le_bytes(pair[0])),
                        f32::from_bits(u32::from_le_bytes(pair[1])),
                    )
                })
                .collect();
            Ok(Value::ComplexTensor(
                ComplexTensor::from_f32(data, shape)
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
            ))
        }
        element_type if complex_integer_element_type(element_type).is_some() => {
            let (scalar_type, width) = complex_integer_element_type(element_type)
                .expect("guarded complex integer element type");
            let pair_width = width * 2;
            if !value.little_endian_data.len().is_multiple_of(pair_width) {
                return Err(ValueCodecError::invalid(
                    path,
                    "complex integer byte length is not element aligned",
                ));
            }
            let mut real = Vec::with_capacity(value.little_endian_data.len() / 2);
            let mut imag = Vec::with_capacity(value.little_endian_data.len() / 2);
            for pair in value.little_endian_data.chunks_exact(pair_width) {
                real.extend_from_slice(&pair[..width]);
                imag.extend_from_slice(&pair[width..]);
            }
            let storage = IntegerComplexStorage::new(
                decode_integer_storage(scalar_type, &real, path)?,
                decode_integer_storage(scalar_type, &imag, path)?,
            )
            .map_err(|error| ValueCodecError::invalid(path, error))?;
            Ok(Value::ComplexTensor(
                ComplexTensor::new_integer(storage, shape)
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
            ))
        }
        element_type => Ok(Value::Tensor(
            Tensor::new_integer(
                decode_integer_storage(element_type, &value.little_endian_data, path)?,
                shape,
            )
            .map_err(|error| ValueCodecError::invalid(path, error))?,
        )),
    }
}

fn decode_sparse(value: &SparseValue, path: &str) -> Result<Value, ValueCodecError> {
    let rows = usize::try_from(value.rows)
        .map_err(|_| ValueCodecError::invalid(path, "row count exceeds platform limits"))?;
    let columns = usize::try_from(value.columns)
        .map_err(|_| ValueCodecError::invalid(path, "column count exceeds platform limits"))?;
    let column_offsets = usize_indices(&value.column_offsets, path)?;
    let row_indices = usize_indices(&value.row_indices, path)?;
    let sparse = match value.element_type {
        ElementType::F64 => SparseTensor::new(
            rows,
            columns,
            column_offsets,
            row_indices,
            decode_f64(&value.little_endian_data, path)?,
        ),
        ElementType::F32 => SparseTensor::new_f32(
            rows,
            columns,
            column_offsets,
            row_indices,
            decode_f32(&value.little_endian_data, path)?,
        ),
        ElementType::Logical
        | ElementType::ComplexF64
        | ElementType::ComplexF32
        | ElementType::ComplexI8
        | ElementType::ComplexI16
        | ElementType::ComplexI32
        | ElementType::ComplexI64
        | ElementType::ComplexU8
        | ElementType::ComplexU16
        | ElementType::ComplexU32
        | ElementType::ComplexU64 => {
            return Err(ValueCodecError::unsupported(
                path,
                "the runtime has no matching sparse value class",
            ))
        }
        element_type => SparseTensor::new_integer(
            rows,
            columns,
            column_offsets,
            row_indices,
            decode_integer_storage(element_type, &value.little_endian_data, path)?,
        ),
    }
    .map_err(|error| ValueCodecError::invalid(path, error))?;
    Ok(Value::SparseTensor(sparse))
}

fn complex_integer_element_type(element_type: ElementType) -> Option<(ElementType, usize)> {
    Some(match element_type {
        ElementType::ComplexI8 => (ElementType::I8, 1),
        ElementType::ComplexI16 => (ElementType::I16, 2),
        ElementType::ComplexI32 => (ElementType::I32, 4),
        ElementType::ComplexI64 => (ElementType::I64, 8),
        ElementType::ComplexU8 => (ElementType::U8, 1),
        ElementType::ComplexU16 => (ElementType::U16, 2),
        ElementType::ComplexU32 => (ElementType::U32, 4),
        ElementType::ComplexU64 => (ElementType::U64, 8),
        _ => return None,
    })
}

fn decode_integer_storage(
    element_type: ElementType,
    bytes: &[u8],
    path: &str,
) -> Result<IntegerStorage, ValueCodecError> {
    Ok(match element_type {
        ElementType::I8 => IntegerStorage::I8(bytes.iter().map(|value| *value as i8).collect()),
        ElementType::I16 => {
            IntegerStorage::I16(decode_signed::<2, i16>(bytes, path, i16::from_le_bytes)?)
        }
        ElementType::I32 => {
            IntegerStorage::I32(decode_signed::<4, i32>(bytes, path, i32::from_le_bytes)?)
        }
        ElementType::I64 => {
            IntegerStorage::I64(decode_signed::<8, i64>(bytes, path, i64::from_le_bytes)?)
        }
        ElementType::U8 => IntegerStorage::U8(bytes.to_vec()),
        ElementType::U16 => {
            IntegerStorage::U16(decode_signed::<2, u16>(bytes, path, u16::from_le_bytes)?)
        }
        ElementType::U32 => {
            IntegerStorage::U32(decode_signed::<4, u32>(bytes, path, u32::from_le_bytes)?)
        }
        ElementType::U64 => {
            IntegerStorage::U64(decode_signed::<8, u64>(bytes, path, u64::from_le_bytes)?)
        }
        _ => {
            return Err(ValueCodecError::invalid(
                path,
                "element type is not an integer",
            ))
        }
    })
}

fn decode_signed<const N: usize, T>(
    bytes: &[u8],
    path: &str,
    decode: fn([u8; N]) -> T,
) -> Result<Vec<T>, ValueCodecError> {
    decode_arrays(bytes, path)?
        .into_iter()
        .map(|word| Ok(decode(word)))
        .collect()
}

fn decode_f32(bytes: &[u8], path: &str) -> Result<Vec<f32>, ValueCodecError> {
    Ok(decode_arrays::<4>(bytes, path)?
        .into_iter()
        .map(|word| f32::from_bits(u32::from_le_bytes(word)))
        .collect())
}

fn decode_f64(bytes: &[u8], path: &str) -> Result<Vec<f64>, ValueCodecError> {
    Ok(decode_arrays::<8>(bytes, path)?
        .into_iter()
        .map(|word| f64::from_bits(u64::from_le_bytes(word)))
        .collect())
}

fn decode_words<const N: usize>(bytes: &[u8], path: &str) -> Result<Vec<u64>, ValueCodecError> {
    if N != 8 {
        return Err(ValueCodecError::invalid(path, "invalid word width"));
    }
    Ok(decode_arrays::<N>(bytes, path)?
        .into_iter()
        .map(|word| {
            let mut full = [0_u8; 8];
            full.copy_from_slice(&word);
            u64::from_le_bytes(full)
        })
        .collect())
}

fn decode_arrays<const N: usize>(
    bytes: &[u8],
    path: &str,
) -> Result<Vec<[u8; N]>, ValueCodecError> {
    if !bytes.len().is_multiple_of(N) {
        return Err(ValueCodecError::invalid(
            path,
            "encoded data is not aligned to its element width",
        ));
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|chunk| chunk.try_into().expect("chunks have exact width"))
        .collect())
}

fn decode_exception(value: &ExceptionValue, path: &str) -> Result<Value, ValueCodecError> {
    if !value.causes.is_empty() {
        return Err(ValueCodecError::unsupported(
            path,
            "nested exception causes are not represented by the current runtime MException",
        ));
    }
    Ok(Value::MException(MException {
        identifier: value.identifier.clone(),
        message: value.message.clone(),
        stack: value.stack.clone(),
    }))
}

fn decode_callable(value: &CallableValue, path: &str) -> Result<Value, ValueCodecError> {
    value
        .validate_identity()
        .map_err(|error| ValueCodecError::invalid(path, error.to_string()))?;
    let captures = value
        .captures
        .iter()
        .enumerate()
        .map(|(index, value)| decode(value, &format!("{path}.capture[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let bound_function = value
        .owner_identity
        .strip_prefix("bound:")
        .map(|value| {
            value
                .parse()
                .map_err(|_| ValueCodecError::invalid(path, "invalid bound callable identity"))
        })
        .transpose()?;
    if !captures.is_empty() {
        return Ok(Value::Closure(Closure {
            function_name: value.qualified_name.clone(),
            bound_function,
            captures,
        }));
    }
    match value.owner_identity.as_str() {
        "workspace" => Ok(Value::FunctionHandle(value.qualified_name.clone())),
        "external" => Ok(Value::ExternalFunctionHandle(value.qualified_name.clone())),
        "method" => Ok(Value::MethodFunctionHandle(value.qualified_name.clone())),
        _ if bound_function.is_some() => Ok(Value::BoundFunctionHandle {
            name: value.qualified_name.clone(),
            function: bound_function.expect("checked above"),
        }),
        _ => Err(ValueCodecError::unsupported(
            path,
            "callable owner is not available in this runtime",
        )),
    }
}

fn usize_shape(shape: &[u64], path: &str) -> Result<Vec<usize>, ValueCodecError> {
    usize_indices(shape, path)
}

fn usize_indices(values: &[u64], path: &str) -> Result<Vec<usize>, ValueCodecError> {
    values
        .iter()
        .map(|value| {
            usize::try_from(*value)
                .map_err(|_| ValueCodecError::invalid(path, "value exceeds platform limits"))
        })
        .collect()
}
