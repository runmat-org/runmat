use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_accelerate_api::IntegerElementType;
use runmat_builtins::{MTIMES_DESCRIPTOR, MTIMES_ERROR_INVALID_INPUT};
use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor};

fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
    err
}

fn integer_scalar_mtimes_cases() -> Vec<(IntegerStorage, IntValue, IntegerStorage)> {
    vec![
        (
            IntegerStorage::I8(vec![i8::MAX, i8::MIN, 2]),
            IntValue::I8(2),
            IntegerStorage::I8(vec![i8::MAX, i8::MIN, 4]),
        ),
        (
            IntegerStorage::I16(vec![i16::MAX, i16::MIN, 2]),
            IntValue::I16(2),
            IntegerStorage::I16(vec![i16::MAX, i16::MIN, 4]),
        ),
        (
            IntegerStorage::I32(vec![i32::MAX, i32::MIN, 2]),
            IntValue::I32(2),
            IntegerStorage::I32(vec![i32::MAX, i32::MIN, 4]),
        ),
        (
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, 2]),
            IntValue::I64(2),
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, 4]),
        ),
        (
            IntegerStorage::U8(vec![u8::MAX, 2, 0]),
            IntValue::U8(2),
            IntegerStorage::U8(vec![u8::MAX, 4, 0]),
        ),
        (
            IntegerStorage::U16(vec![u16::MAX, 2, 0]),
            IntValue::U16(2),
            IntegerStorage::U16(vec![u16::MAX, 4, 0]),
        ),
        (
            IntegerStorage::U32(vec![u32::MAX, 2, 0]),
            IntValue::U32(2),
            IntegerStorage::U32(vec![u32::MAX, 4, 0]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 53) + 1, 0]),
            IntValue::U64(2),
            IntegerStorage::U64(vec![u64::MAX, (1_u64 << 54) + 2, 0]),
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

mod host;
mod integer;
mod provider;
mod wgpu;

fn mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::mtimes_builtin(lhs, rhs))
}
