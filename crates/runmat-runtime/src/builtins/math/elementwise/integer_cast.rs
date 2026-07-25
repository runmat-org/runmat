//! Shared exact host-side conversion support for MATLAB integer cast builtins.

use runmat_builtins::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor, Value,
};

use crate::builtins::common::tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntegerTarget {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntegerTarget {
    pub(crate) fn class_name(self) -> &'static str {
        match self {
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
        }
    }

    pub(crate) fn cast_i128(self, value: i128) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8),
            Self::I16 => IntValue::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16),
            Self::I32 => IntValue::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32),
            Self::I64 => IntValue::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64),
            Self::U8 => IntValue::U8(value.clamp(0, u8::MAX as i128) as u8),
            Self::U16 => IntValue::U16(value.clamp(0, u16::MAX as i128) as u16),
            Self::U32 => IntValue::U32(value.clamp(0, u32::MAX as i128) as u32),
            Self::U64 => IntValue::U64(value.clamp(0, u64::MAX as i128) as u64),
        }
    }

    pub(crate) fn uses_extended_scalar_precision(self) -> bool {
        matches!(self, Self::I64 | Self::U64)
    }

    pub(crate) fn from_int_value(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => Self::I8,
            IntValue::I16(_) => Self::I16,
            IntValue::I32(_) => Self::I32,
            IntValue::I64(_) => Self::I64,
            IntValue::U8(_) => Self::U8,
            IntValue::U16(_) => Self::U16,
            IntValue::U32(_) => Self::U32,
            IntValue::U64(_) => Self::U64,
        }
    }

    pub(crate) fn from_storage(storage: &IntegerStorage) -> Self {
        match storage {
            IntegerStorage::I8(_) => Self::I8,
            IntegerStorage::I16(_) => Self::I16,
            IntegerStorage::I32(_) => Self::I32,
            IntegerStorage::I64(_) => Self::I64,
            IntegerStorage::U8(_) => Self::U8,
            IntegerStorage::U16(_) => Self::U16,
            IntegerStorage::U32(_) => Self::U32,
            IntegerStorage::U64(_) => Self::U64,
        }
    }

    pub(crate) fn accelerator_type(self) -> runmat_accelerate_api::IntegerElementType {
        match self {
            Self::I8 => runmat_accelerate_api::IntegerElementType::I8,
            Self::I16 => runmat_accelerate_api::IntegerElementType::I16,
            Self::I32 => runmat_accelerate_api::IntegerElementType::I32,
            Self::I64 => runmat_accelerate_api::IntegerElementType::I64,
            Self::U8 => runmat_accelerate_api::IntegerElementType::U8,
            Self::U16 => runmat_accelerate_api::IntegerElementType::U16,
            Self::U32 => runmat_accelerate_api::IntegerElementType::U32,
            Self::U64 => runmat_accelerate_api::IntegerElementType::U64,
        }
    }

    pub(crate) fn cast_scalar(self, value: f64) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(cast_signed(value, i8::MIN as f64, i8::MAX as f64) as i8),
            Self::I16 => IntValue::I16(cast_signed(value, i16::MIN as f64, i16::MAX as f64) as i16),
            Self::I32 => IntValue::I32(cast_signed(value, i32::MIN as f64, i32::MAX as f64) as i32),
            Self::I64 => IntValue::I64(cast_signed(value, i64::MIN as f64, i64::MAX as f64) as i64),
            Self::U8 => IntValue::U8(cast_unsigned(value, u8::MAX as f64) as u8),
            Self::U16 => IntValue::U16(cast_unsigned(value, u16::MAX as f64) as u16),
            Self::U32 => IntValue::U32(cast_unsigned(value, u32::MAX as f64) as u32),
            Self::U64 => IntValue::U64(cast_unsigned(value, u64::MAX as f64) as u64),
        }
    }

    pub(crate) fn cast_int(self, value: &IntValue) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8),
            Self::I16 => {
                IntValue::I16(value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::I32 => {
                IntValue::I32(value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::I64 => IntValue::I64(value.to_i64()),
            Self::U8 => IntValue::U8(unsigned_value(value).min(u8::MAX as u64) as u8),
            Self::U16 => IntValue::U16(unsigned_value(value).min(u16::MAX as u64) as u16),
            Self::U32 => IntValue::U32(unsigned_value(value).min(u32::MAX as u64) as u32),
            Self::U64 => IntValue::U64(unsigned_value(value)),
        }
    }

    pub(crate) fn cast_tensor(self, tensor: Tensor) -> Result<Tensor, String> {
        let values = match tensor.integer_data {
            Some(storage) => integer_values(storage)
                .iter()
                .map(|value| self.cast_int(value))
                .collect(),
            None => tensor
                .data
                .iter()
                .map(|&value| self.cast_scalar(value))
                .collect(),
        };
        Tensor::new_integer(self.storage(values), tensor.shape)
    }

    pub(crate) fn storage(self, values: Vec<IntValue>) -> IntegerStorage {
        match self {
            Self::I8 => IntegerStorage::I8(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I8(value) => value,
                        _ => unreachable!("target conversion must produce int8"),
                    })
                    .collect(),
            ),
            Self::I16 => IntegerStorage::I16(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I16(value) => value,
                        _ => unreachable!("target conversion must produce int16"),
                    })
                    .collect(),
            ),
            Self::I32 => IntegerStorage::I32(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I32(value) => value,
                        _ => unreachable!("target conversion must produce int32"),
                    })
                    .collect(),
            ),
            Self::I64 => IntegerStorage::I64(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I64(value) => value,
                        _ => unreachable!("target conversion must produce int64"),
                    })
                    .collect(),
            ),
            Self::U8 => IntegerStorage::U8(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U8(value) => value,
                        _ => unreachable!("target conversion must produce uint8"),
                    })
                    .collect(),
            ),
            Self::U16 => IntegerStorage::U16(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U16(value) => value,
                        _ => unreachable!("target conversion must produce uint16"),
                    })
                    .collect(),
            ),
            Self::U32 => IntegerStorage::U32(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U32(value) => value,
                        _ => unreachable!("target conversion must produce uint32"),
                    })
                    .collect(),
            ),
            Self::U64 => IntegerStorage::U64(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U64(value) => value,
                        _ => unreachable!("target conversion must produce uint64"),
                    })
                    .collect(),
            ),
        }
    }
}

#[derive(Debug)]
pub(crate) enum CastError {
    Unsupported(String),
    Internal(String),
}

pub(crate) async fn cast_value(value: Value, target: IntegerTarget) -> Result<Value, CastError> {
    match value {
        Value::Num(value) => Ok(Value::Int(target.cast_scalar(value))),
        Value::Int(value) => Ok(Value::Int(target.cast_int(&value))),
        Value::Bool(value) => Ok(Value::Int(target.cast_scalar(if value {
            1.0
        } else {
            0.0
        }))),
        Value::Tensor(tensor) => cast_tensor_value(target, tensor),
        Value::SparseTensor(sparse) => cast_sparse_value(target, sparse),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array).map_err(CastError::Internal)?;
            cast_tensor_value(target, tensor)
        }
        Value::CharArray(chars) => {
            let tensor = Tensor::new(
                chars
                    .data
                    .iter()
                    .map(|&value| value as u32 as f64)
                    .collect(),
                vec![chars.rows, chars.cols],
            )
            .map_err(CastError::Internal)?;
            cast_tensor_value(target, tensor)
        }
        Value::GpuTensor(handle) => {
            let provider = runmat_accelerate_api::provider()
                .ok_or_else(|| CastError::Internal("no acceleration provider registered".into()))?;
            provider
                .cast_to_integer(&handle, target.accelerator_type())
                .await
                .map(Value::GpuTensor)
                .map_err(|error| CastError::Internal(error.to_string()))
        }
        value @ (Value::Complex(_, _) | Value::ComplexTensor(_)) => {
            cast_complex_value(value, target)
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(CastError::Unsupported("string".to_string()))
        }
        Value::Symbolic(expression) => expression
            .numeric_constant_value()
            .map(|value| Value::Int(target.cast_scalar(value)))
            .ok_or_else(|| CastError::Unsupported("sym".to_string())),
        Value::Cell(_) => Err(CastError::Unsupported("cell".to_string())),
        Value::Struct(_) => Err(CastError::Unsupported("struct".to_string())),
        Value::Object(object) => Err(CastError::Unsupported(object.class_name)),
        Value::HandleObject(handle) => Err(CastError::Unsupported(handle.class_name)),
        Value::Listener(_) => Err(CastError::Unsupported("event.listener".to_string())),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(CastError::Unsupported("function_handle".to_string())),
        Value::ClassRef(_) => Err(CastError::Unsupported("meta.class".to_string())),
        Value::MException(_) => Err(CastError::Unsupported("MException".to_string())),
        Value::OutputList(_) => Err(CastError::Unsupported("OutputList".to_string())),
    }
}

pub(crate) fn cast_sparse_value(
    target: IntegerTarget,
    sparse: runmat_builtins::SparseTensor,
) -> Result<Value, CastError> {
    let values = match sparse.integer_storage() {
        Some(storage) => storage
            .exact_values()
            .iter()
            .map(|value| target.cast_int(value))
            .collect(),
        None => sparse
            .values
            .iter()
            .map(|&value| target.cast_scalar(value))
            .collect(),
    };
    let storage = target.storage(values);
    runmat_builtins::SparseTensor::new_integer(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs,
        sparse.row_indices,
        storage,
    )
    .map(Value::SparseTensor)
    .map_err(CastError::Internal)
}

pub(crate) fn cast_complex_value(value: Value, target: IntegerTarget) -> Result<Value, CastError> {
    let (real, imag, shape) = match value {
        Value::Complex(real, imag) => (
            vec![target.cast_scalar(real)],
            vec![target.cast_scalar(imag)],
            vec![1, 1],
        ),
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape;
            if let Some(storage) = tensor.integer_data {
                (
                    integer_values(storage.real)
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                    integer_values(storage.imag)
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                    shape,
                )
            } else {
                let (real, imag): (Vec<_>, Vec<_>) = tensor
                    .data
                    .into_iter()
                    .map(|(real, imag)| (target.cast_scalar(real), target.cast_scalar(imag)))
                    .unzip();
                (real, imag, shape)
            }
        }
        _ => return Err(CastError::Unsupported("complex".to_string())),
    };

    let storage = IntegerComplexStorage::new(target.storage(real), target.storage(imag))
        .map_err(CastError::Internal)?;
    ComplexTensor::new_integer(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(CastError::Internal)
}

fn cast_tensor_value(target: IntegerTarget, tensor: Tensor) -> Result<Value, CastError> {
    let tensor = target.cast_tensor(tensor).map_err(CastError::Internal)?;
    if tensor.data.len() == 1 {
        let storage = tensor
            .integer_data
            .expect("integer cast must construct exact integer storage");
        Ok(Value::Int(
            integer_values(storage).pop().expect("scalar storage"),
        ))
    } else {
        Ok(Value::Tensor(tensor))
    }
}

fn cast_signed(value: f64, min: f64, max: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            min as i64
        } else {
            max as i64
        }
    } else {
        value.round().clamp(min, max) as i64
    }
}

fn cast_unsigned(value: f64, max: f64) -> u64 {
    if value.is_nan() || value.is_sign_negative() {
        0
    } else if value.is_infinite() {
        u64::MAX.min(max as u64)
    } else {
        value.round().clamp(0.0, max) as u64
    }
}

fn unsigned_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U64(value) => *value,
        _ => value.to_i64().max(0) as u64,
    }
}

pub(crate) fn integer_values(storage: IntegerStorage) -> Vec<IntValue> {
    match storage {
        IntegerStorage::I8(values) => values.into_iter().map(IntValue::I8).collect(),
        IntegerStorage::I16(values) => values.into_iter().map(IntValue::I16).collect(),
        IntegerStorage::I32(values) => values.into_iter().map(IntValue::I32).collect(),
        IntegerStorage::I64(values) => values.into_iter().map(IntValue::I64).collect(),
        IntegerStorage::U8(values) => values.into_iter().map(IntValue::U8).collect(),
        IntegerStorage::U16(values) => values.into_iter().map(IntValue::U16).collect(),
        IntegerStorage::U32(values) => values.into_iter().map(IntValue::U32).collect(),
        IntegerStorage::U64(values) => values.into_iter().map(IntValue::U64).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn uint64_to_int64_array_saturates_without_f64_rounding() {
        let source = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("source tensor");
        let output = IntegerTarget::I64
            .cast_tensor(source)
            .expect("int64 conversion");

        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MAX]))
        );
    }

    #[test]
    fn int64_to_uint64_array_clamps_negative_values_exactly() {
        let source = Tensor::new_integer(IntegerStorage::I64(vec![-1, i64::MAX]), vec![1, 2])
            .expect("source tensor");
        let output = IntegerTarget::U64
            .cast_tensor(source)
            .expect("uint64 conversion");

        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, i64::MAX as u64]))
        );
    }

    #[test]
    fn sparse_casts_preserve_structure_and_convert_every_integer_class() {
        let sparse =
            runmat_builtins::SparseTensor::new(3, 2, vec![0, 1, 2], vec![0, 2], vec![1.5, -2.5])
                .expect("sparse input");
        let cases = [
            (IntegerTarget::I8, "int8", vec![2.0, -3.0]),
            (IntegerTarget::I16, "int16", vec![2.0, -3.0]),
            (IntegerTarget::I32, "int32", vec![2.0, -3.0]),
            (IntegerTarget::I64, "int64", vec![2.0, -3.0]),
            (IntegerTarget::U8, "uint8", vec![2.0, 0.0]),
            (IntegerTarget::U16, "uint16", vec![2.0, 0.0]),
            (IntegerTarget::U32, "uint32", vec![2.0, 0.0]),
            (IntegerTarget::U64, "uint64", vec![2.0, 0.0]),
        ];
        for (target, class, expected) in cases {
            let Value::SparseTensor(output) =
                cast_sparse_value(target, sparse.clone()).expect("sparse cast")
            else {
                panic!("integer cast must retain sparse storage");
            };
            assert_eq!(output.shape(), vec![3, 2]);
            assert_eq!(output.col_ptrs, vec![0, 1, 2]);
            assert_eq!(output.row_indices, vec![0, 2]);
            assert_eq!(output.values, expected);
            assert_eq!(
                output.integer_storage().map(IntegerStorage::class_name),
                Some(class)
            );
        }

        let exact = runmat_builtins::SparseTensor::new_integer(
            1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("exact sparse input");
        let Value::SparseTensor(output) =
            cast_sparse_value(IntegerTarget::I64, exact).expect("exact sparse cast")
        else {
            panic!("integer cast must retain sparse storage");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MAX]))
        );
    }

    #[test]
    fn every_integer_cast_builtin_dispatches_sparse_inputs() {
        let sparse = runmat_builtins::SparseTensor::new(2, 1, vec![0, 1], vec![1], vec![1.5])
            .expect("sparse input");
        for (builtin, class) in [
            ("int8", "int8"),
            ("int16", "int16"),
            ("int32", "int32"),
            ("int64", "int64"),
            ("uint8", "uint8"),
            ("uint16", "uint16"),
            ("uint32", "uint32"),
            ("uint64", "uint64"),
        ] {
            let Value::SparseTensor(output) =
                crate::dispatcher::call_builtin(builtin, &[Value::SparseTensor(sparse.clone())])
                    .expect("integer cast dispatch")
            else {
                panic!("{builtin} must preserve sparse storage");
            };
            assert_eq!(
                output.integer_storage().map(IntegerStorage::class_name),
                Some(class)
            );
            assert_eq!(output.get(1, 0), Some(2.0));
        }
    }

    #[test]
    fn integer_cast_builtins_dispatch_exact_typed_tensor_inputs() {
        let cases = [
            (
                "int32",
                IntegerStorage::U64(vec![u64::MAX, 1]),
                IntegerStorage::I32(vec![i32::MAX, 1]),
            ),
            (
                "uint8",
                IntegerStorage::I64(vec![-1, 300]),
                IntegerStorage::U8(vec![0, u8::MAX]),
            ),
            (
                "uint16",
                IntegerStorage::I64(vec![-1, 70_000]),
                IntegerStorage::U16(vec![0, u16::MAX]),
            ),
            (
                "uint32",
                IntegerStorage::U64(vec![u64::MAX, 1]),
                IntegerStorage::U32(vec![u32::MAX, 1]),
            ),
        ];

        for (builtin, input_storage, expected_storage) in cases {
            let input = Tensor::new_integer(input_storage, vec![1, 2]).expect("typed input");
            let Value::Tensor(output) =
                crate::dispatcher::call_builtin(builtin, &[Value::Tensor(input)])
                    .expect("integer cast dispatch")
            else {
                panic!("{builtin} must return a typed tensor");
            };
            assert_eq!(output.integer_storage(), Some(&expected_storage));
        }
    }

    #[test]
    fn complex_float_arrays_convert_every_integer_class_and_remain_complex() {
        for target in [
            IntegerTarget::I8,
            IntegerTarget::I16,
            IntegerTarget::I32,
            IntegerTarget::I64,
            IntegerTarget::U8,
            IntegerTarget::U16,
            IntegerTarget::U32,
            IntegerTarget::U64,
        ] {
            let input = ComplexTensor::new(vec![(1.5, 0.49), (-2.5, -1.5)], vec![1, 2])
                .expect("complex input");
            let output = block_on(cast_value(Value::ComplexTensor(input), target))
                .expect("complex integer conversion");
            let Value::ComplexTensor(output) = output else {
                panic!("integer conversion must preserve complex storage");
            };
            let expected = IntegerComplexStorage::new(
                target.storage(vec![target.cast_scalar(1.5), target.cast_scalar(-2.5)]),
                target.storage(vec![target.cast_scalar(0.49), target.cast_scalar(-1.5)]),
            )
            .expect("matching storage");
            assert_eq!(output.shape, vec![1, 2]);
            assert_eq!(output.integer_data, Some(expected));
        }
    }

    #[test]
    fn typed_complex_casts_preserve_exact_uint64_components_before_saturation() {
        let input = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                IntegerStorage::U64(vec![1, 2]),
            )
            .expect("matching components"),
            vec![1, 2],
        )
        .expect("typed complex input");
        let output = block_on(cast_value(Value::ComplexTensor(input), IntegerTarget::I64))
            .expect("int64 conversion");
        let Value::ComplexTensor(output) = output else {
            panic!("integer conversion must preserve complex storage");
        };
        assert_eq!(
            output.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MAX, i64::MAX]),
                    IntegerStorage::I64(vec![1, 2]),
                )
                .expect("matching components")
            )
        );
    }
}
