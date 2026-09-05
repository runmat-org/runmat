use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_builtins::{CONJ_DESCRIPTOR, CONJ_ERROR_INVALID_INPUT};
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor, Value,
};

use super::*;

fn conj_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::conj_builtin(value))
}

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}

mod host;
mod provider;
#[cfg(feature = "wgpu")]
mod wgpu;
