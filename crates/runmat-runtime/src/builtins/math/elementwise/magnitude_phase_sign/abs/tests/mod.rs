use super::host::*;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::{ABS_DESCRIPTOR, ABS_EXTENSIONS, ABS_INTEGER_CAPABILITIES};

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}
use runmat_value::{IntValue, IntegerComplexStorage, LogicalArray, Tensor};

fn abs_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::abs_builtin(value))
}

mod provider;
mod semantics;
mod values;
