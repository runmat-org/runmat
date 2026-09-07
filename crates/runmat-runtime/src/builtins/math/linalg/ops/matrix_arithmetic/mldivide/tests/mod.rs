use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use nalgebra::DMatrix;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::IntegerElementType;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderTelemetry};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerOutputClassRule, MLDIVIDE_DESCRIPTOR,
    MLDIVIDE_ERROR_INVALID_INPUT,
};
use runmat_value::{ComplexTensor, IntValue, IntegerStorage};
fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
    err
}

fn fallback_count(telemetry: &ProviderTelemetry, reason: &str) -> u64 {
    telemetry
        .solve_fallbacks
        .iter()
        .find(|entry| entry.reason == reason)
        .map(|entry| entry.count)
        .unwrap_or(0)
}

fn clear_accel_provider_state() {
    runmat_accelerate_api::set_thread_provider(None);
    runmat_accelerate_api::clear_provider();
}

#[cfg(feature = "wgpu")]
fn integer_scalar_mldivide_cases() -> Vec<(IntValue, IntegerStorage, IntegerStorage)> {
    vec![
        (
            IntValue::I8(2),
            IntegerStorage::I8(vec![6, -8, 10]),
            IntegerStorage::I8(vec![3, -4, 5]),
        ),
        (
            IntValue::I16(2),
            IntegerStorage::I16(vec![6, -8, 10]),
            IntegerStorage::I16(vec![3, -4, 5]),
        ),
        (
            IntValue::I32(2),
            IntegerStorage::I32(vec![6, -8, 10]),
            IntegerStorage::I32(vec![3, -4, 5]),
        ),
        (
            IntValue::I64(2),
            IntegerStorage::I64(vec![6, -8, 10]),
            IntegerStorage::I64(vec![3, -4, 5]),
        ),
        (
            IntValue::U8(2),
            IntegerStorage::U8(vec![6, 8, 10]),
            IntegerStorage::U8(vec![3, 4, 5]),
        ),
        (
            IntValue::U16(2),
            IntegerStorage::U16(vec![6, 8, 10]),
            IntegerStorage::U16(vec![3, 4, 5]),
        ),
        (
            IntValue::U32(2),
            IntegerStorage::U32(vec![6, 8, 10]),
            IntegerStorage::U32(vec![3, 4, 5]),
        ),
        (
            IntValue::U64(2),
            IntegerStorage::U64(vec![6, 8, 10]),
            IntegerStorage::U64(vec![3, 4, 5]),
        ),
    ]
}

#[cfg(feature = "wgpu")]
fn integer_element_type(storage: &IntegerStorage) -> IntegerElementType {
    match storage {
        IntegerStorage::I8(_) => IntegerElementType::I8,
        IntegerStorage::I16(_) => IntegerElementType::I16,
        IntegerStorage::I32(_) => IntegerElementType::I32,
        IntegerStorage::I64(_) => IntegerElementType::I64,
        IntegerStorage::U8(_) => IntegerElementType::U8,
        IntegerStorage::U16(_) => IntegerElementType::U16,
        IntegerStorage::U32(_) => IntegerElementType::U32,
        IntegerStorage::U64(_) => IntegerElementType::U64,
    }
}

use num_complex::Complex64;

mod contract;
mod host;
mod integer;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;

fn mldivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::mldivide_builtin(lhs, rhs))
}

fn mldivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    block_on(super::mldivide_eval(lhs, rhs))
}
