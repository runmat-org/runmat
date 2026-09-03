use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{PRIMES_ERROR_INTERNAL, PRIMES_ERROR_INVALID_INPUT};
use runmat_types::NumericClass;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

use super::error::primes_error;
use crate::builtins::common::{gpu_helpers, tensor as tensor_utils};
use crate::BuiltinResult;

pub(super) const MAX_PRIMES_LIMIT: u64 = 10_000_000;

#[derive(Clone, Copy, Debug)]
pub(super) struct PrimeRequest {
    pub(super) limit: u64,
    pub(super) output_class: NumericClass,
}

pub(super) async fn parse_request(value: Value) -> BuiltinResult<PrimeRequest> {
    match value {
        Value::Num(value) => from_float(value, NumericClass::Double),
        Value::Int(value) => Ok(from_integer(value)),
        Value::Tensor(tensor) => from_tensor(tensor),
        Value::GpuTensor(handle) => from_gpu(handle).await,
        Value::Bool(_) | Value::LogicalArray(_) => Err(primes_error(
            &PRIMES_ERROR_INVALID_INPUT,
            "n must be numeric, not logical",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(primes_error(&PRIMES_ERROR_INVALID_INPUT, "n must be real"))
        }
        other => Err(primes_error(
            &PRIMES_ERROR_INVALID_INPUT,
            format!("unsupported input type {other:?}"),
        )),
    }
}

async fn from_gpu(handle: GpuTensorHandle) -> BuiltinResult<PrimeRequest> {
    if handle
        .shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        != Some(1)
    {
        return Err(primes_error(
            &PRIMES_ERROR_INVALID_INPUT,
            "n must be a scalar value",
        ));
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|error| primes_error(&PRIMES_ERROR_INTERNAL, error))?;
    from_tensor(tensor)
}

fn from_tensor(tensor: Tensor) -> BuiltinResult<PrimeRequest> {
    if !tensor_utils::is_scalar_tensor(&tensor) {
        return Err(primes_error(
            &PRIMES_ERROR_INVALID_INPUT,
            "n must be a scalar value",
        ));
    }
    if let Some(storage) = tensor.integer_storage() {
        return Ok(from_integer(
            storage
                .value_at(0)
                .expect("scalar integer storage contains one value"),
        ));
    }
    let output_class = match tensor.numeric_dtype() {
        NumericDType::F32 => NumericClass::Single,
        NumericDType::F64 => NumericClass::Double,
        _ => unreachable!("integer tensor dtype has authoritative integer storage"),
    };
    from_float(tensor_utils::tensor_value_f64(&tensor, 0), output_class)
}

fn from_integer(value: IntValue) -> PrimeRequest {
    let (limit, output_class) = match value {
        IntValue::I8(value) => (signed_limit(i64::from(value)), NumericClass::Int8),
        IntValue::I16(value) => (signed_limit(i64::from(value)), NumericClass::Int16),
        IntValue::I32(value) => (signed_limit(i64::from(value)), NumericClass::Int32),
        IntValue::I64(value) => (signed_limit(value), NumericClass::Int64),
        IntValue::U8(value) => (u64::from(value), NumericClass::UInt8),
        IntValue::U16(value) => (u64::from(value), NumericClass::UInt16),
        IntValue::U32(value) => (u64::from(value), NumericClass::UInt32),
        IntValue::U64(value) => (value, NumericClass::UInt64),
    };
    PrimeRequest {
        limit,
        output_class,
    }
}

fn signed_limit(value: i64) -> u64 {
    if value < 2 {
        0
    } else {
        value as u64
    }
}

fn from_float(value: f64, output_class: NumericClass) -> BuiltinResult<PrimeRequest> {
    if !value.is_finite() {
        return Err(primes_error(
            &PRIMES_ERROR_INVALID_INPUT,
            "n must be finite",
        ));
    }
    if value.fract() != 0.0 {
        return Err(primes_error(
            &PRIMES_ERROR_INVALID_INPUT,
            "n must be an integer value",
        ));
    }
    let limit = if value < 2.0 {
        0
    } else if value >= u64::MAX as f64 {
        return Err(primes_error(&PRIMES_ERROR_INVALID_INPUT, "n is too large"));
    } else {
        value as u64
    };
    Ok(PrimeRequest {
        limit,
        output_class,
    })
}
