use runmat_types::NumericClass;
use runmat_value::{IntValue, NumericStorage, Tensor, Value};

use super::{binary_error, BinaryContext};
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

pub(in crate::builtins::math::discrete) struct BinaryInput {
    pub(in crate::builtins::math::discrete) data: Vec<u128>,
    pub(in crate::builtins::math::discrete) negative: Vec<bool>,
    pub(in crate::builtins::math::discrete) shape: Vec<usize>,
    pub(in crate::builtins::math::discrete) class: NumericClass,
}

impl BinaryInput {
    pub(in crate::builtins::math::discrete) fn is_scalar(&self) -> bool {
        self.data.len() == 1 && self.shape.iter().copied().product::<usize>() == 1
    }

    pub(in crate::builtins::math::discrete) async fn from_value(
        value: Value,
        context: &'static BinaryContext,
    ) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Self::from_float(value, NumericClass::Double, context),
            Value::Int(value) => Self::from_integer(value, context),
            Value::Tensor(tensor) => Self::from_tensor(tensor, context),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|error| binary_error(context, context.internal, error))?;
                Self::from_tensor(tensor, context)
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(binary_error(
                context,
                context.invalid,
                "inputs must be real",
            )),
            Value::Bool(_) | Value::LogicalArray(_) => Err(binary_error(
                context,
                context.invalid,
                "logical inputs are not numeric integer classes",
            )),
            other => Err(binary_error(
                context,
                context.invalid,
                format!("unsupported input type {other:?}"),
            )),
        }
    }

    fn from_float(
        value: f64,
        class: NumericClass,
        context: &'static BinaryContext,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            data: vec![magnitude_from_float(value, context)?],
            negative: vec![value < 0.0],
            shape: vec![1, 1],
            class,
        })
    }

    fn from_integer(value: IntValue, context: &'static BinaryContext) -> BuiltinResult<Self> {
        let negative = integer_is_negative(&value);
        let (magnitude, class) = match value {
            IntValue::I8(value) => (
                magnitude_from_signed(i128::from(value), context)?,
                NumericClass::Int8,
            ),
            IntValue::I16(value) => (
                magnitude_from_signed(i128::from(value), context)?,
                NumericClass::Int16,
            ),
            IntValue::I32(value) => (
                magnitude_from_signed(i128::from(value), context)?,
                NumericClass::Int32,
            ),
            IntValue::I64(value) => (
                magnitude_from_signed(i128::from(value), context)?,
                NumericClass::Int64,
            ),
            IntValue::U8(value) => (
                magnitude_from_unsigned(u128::from(value), context)?,
                NumericClass::UInt8,
            ),
            IntValue::U16(value) => (
                magnitude_from_unsigned(u128::from(value), context)?,
                NumericClass::UInt16,
            ),
            IntValue::U32(value) => (
                magnitude_from_unsigned(u128::from(value), context)?,
                NumericClass::UInt32,
            ),
            IntValue::U64(value) => (
                magnitude_from_unsigned(u128::from(value), context)?,
                NumericClass::UInt64,
            ),
        };
        Ok(Self {
            data: vec![magnitude],
            negative: vec![negative],
            shape: vec![1, 1],
            class,
        })
    }

    fn from_tensor(tensor: Tensor, context: &'static BinaryContext) -> BuiltinResult<Self> {
        let shape = tensor.shape.clone();
        let storage = tensor
            .into_numeric_storage()
            .map_err(|error| binary_error(context, context.internal, error))?;
        let negative = negative_flags(&storage);
        let (data, class) = storage_magnitudes(storage, context)?;
        Ok(Self {
            data,
            negative,
            shape,
            class,
        })
    }
}

fn negative_flags(storage: &NumericStorage) -> Vec<bool> {
    macro_rules! flags {
        ($values:expr, $zero:expr) => {
            $values.iter().map(|&value| value < $zero).collect()
        };
    }
    match storage {
        NumericStorage::F64(values) => flags!(values, 0.0),
        NumericStorage::F32(values) => flags!(values, 0.0),
        NumericStorage::I8(values) => flags!(values, 0),
        NumericStorage::I16(values) => flags!(values, 0),
        NumericStorage::I32(values) => flags!(values, 0),
        NumericStorage::I64(values) => flags!(values, 0),
        NumericStorage::U8(values) => vec![false; values.len()],
        NumericStorage::U16(values) => vec![false; values.len()],
        NumericStorage::U32(values) => vec![false; values.len()],
        NumericStorage::U64(values) => vec![false; values.len()],
    }
}

fn storage_magnitudes(
    storage: NumericStorage,
    context: &'static BinaryContext,
) -> BuiltinResult<(Vec<u128>, NumericClass)> {
    macro_rules! signed {
        ($values:expr, $class:expr) => {
            (
                $values
                    .into_iter()
                    .map(|value| magnitude_from_signed(i128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                $class,
            )
        };
    }
    macro_rules! unsigned {
        ($values:expr, $class:expr) => {
            (
                $values
                    .into_iter()
                    .map(|value| magnitude_from_unsigned(u128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                $class,
            )
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => (
            values
                .into_iter()
                .map(|value| magnitude_from_float(value, context))
                .collect::<BuiltinResult<Vec<_>>>()?,
            NumericClass::Double,
        ),
        NumericStorage::F32(values) => (
            values
                .into_iter()
                .map(|value| magnitude_from_float(f64::from(value), context))
                .collect::<BuiltinResult<Vec<_>>>()?,
            NumericClass::Single,
        ),
        NumericStorage::I8(values) => signed!(values, NumericClass::Int8),
        NumericStorage::I16(values) => signed!(values, NumericClass::Int16),
        NumericStorage::I32(values) => signed!(values, NumericClass::Int32),
        NumericStorage::I64(values) => signed!(values, NumericClass::Int64),
        NumericStorage::U8(values) => unsigned!(values, NumericClass::UInt8),
        NumericStorage::U16(values) => unsigned!(values, NumericClass::UInt16),
        NumericStorage::U32(values) => unsigned!(values, NumericClass::UInt32),
        NumericStorage::U64(values) => unsigned!(values, NumericClass::UInt64),
    })
}

fn integer_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

fn magnitude_from_signed(value: i128, context: &'static BinaryContext) -> BuiltinResult<u128> {
    if !context.accepts_zero_or_negative && value <= 0 {
        return Err(binary_error(
            context,
            context.invalid,
            "inputs must be positive integers",
        ));
    }
    Ok(value.unsigned_abs())
}

fn magnitude_from_unsigned(value: u128, context: &'static BinaryContext) -> BuiltinResult<u128> {
    if !context.accepts_zero_or_negative && value == 0 {
        return Err(binary_error(
            context,
            context.invalid,
            "inputs must be positive integers",
        ));
    }
    Ok(value)
}

fn magnitude_from_float(value: f64, context: &'static BinaryContext) -> BuiltinResult<u128> {
    let invalid_domain = !value.is_finite()
        || value.fract() != 0.0
        || (!context.accepts_zero_or_negative && value <= 0.0);
    if invalid_domain {
        let detail = if context.accepts_zero_or_negative {
            "inputs must be finite integer values"
        } else {
            "inputs must be finite positive integers"
        };
        return Err(binary_error(context, context.invalid, detail));
    }
    if value.abs() >= u64::MAX as f64 {
        return Err(binary_error(context, context.invalid, "input is too large"));
    }
    Ok(value.abs() as u128)
}
