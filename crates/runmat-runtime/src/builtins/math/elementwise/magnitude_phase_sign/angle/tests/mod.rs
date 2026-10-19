use super::host::*;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::BuiltinIntegerInputAvailability;

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}
use runmat_value::{CharArray, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray};
use std::f64::consts::PI;

fn angle_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::angle_builtin(value))
}

fn all_integer_storages() -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ]
}

mod host;
mod provider;
