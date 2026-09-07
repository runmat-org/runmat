use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, HostTensorView, IntegerElementType, ProviderTelemetry};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerOutputClassRule, MRDIVIDE_DESCRIPTOR,
    MRDIVIDE_ERROR_INVALID_INPUT,
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

fn host_mrdivide_real(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    super::mrdivide_host_real_for_provider(lhs, rhs).expect("host mrdivide")
}

fn clear_accel_provider_state() {
    runmat_accelerate_api::set_thread_provider(None);
    runmat_accelerate_api::clear_provider();
}

fn integer_scalar_mrdivide_cases() -> Vec<(IntegerStorage, IntValue, IntegerStorage)> {
    vec![
        (
            IntegerStorage::I8(vec![i8::MIN, 6, 4]),
            IntValue::I8(2),
            IntegerStorage::I8(vec![i8::MIN / 2, 3, 2]),
        ),
        (
            IntegerStorage::I16(vec![i16::MIN, 6, 4]),
            IntValue::I16(2),
            IntegerStorage::I16(vec![i16::MIN / 2, 3, 2]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, 6, 4]),
            IntValue::I32(2),
            IntegerStorage::I32(vec![i32::MIN / 2, 3, 2]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, 6, 4]),
            IntValue::I64(2),
            IntegerStorage::I64(vec![i64::MIN / 2, 3, 2]),
        ),
        (
            IntegerStorage::U8(vec![u8::MAX - 1, 6, 4]),
            IntValue::U8(2),
            IntegerStorage::U8(vec![(u8::MAX - 1) / 2, 3, 2]),
        ),
        (
            IntegerStorage::U16(vec![u16::MAX - 1, 6, 4]),
            IntValue::U16(2),
            IntegerStorage::U16(vec![(u16::MAX - 1) / 2, 3, 2]),
        ),
        (
            IntegerStorage::U32(vec![u32::MAX - 1, 6, 4]),
            IntValue::U32(2),
            IntegerStorage::U32(vec![(u32::MAX - 1) / 2, 3, 2]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX - 1, (1_u64 << 53) + 2, 4]),
            IntValue::U64(2),
            IntegerStorage::U64(vec![(u64::MAX - 1) / 2, (1_u64 << 52) + 1, 2]),
        ),
    ]
}

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

mod contract;
mod host;
mod integer;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;

fn mrdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::mrdivide_builtin(lhs, rhs))
}

fn mrdivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    block_on(super::mrdivide_eval(lhs, rhs))
}
