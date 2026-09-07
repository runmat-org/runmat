use super::*;
use futures::executor::block_on;
use runmat_builtins::{BSXFUN_ERROR_FUNCTION_ERROR, BSXFUN_ERROR_SIZE_MISMATCH};
use runmat_value::{
    ComplexTensor, IntValue, IntegerStorage, NumericScalar, NumericStorage, Tensor,
};
use std::sync::Arc;

fn call(function: Value, left: Value, right: Value) -> BuiltinResult<Value> {
    block_on(bsxfun_builtin(function, left, right))
}

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}

mod broadcast;
mod invocation;
mod numeric;
mod output_contract;
#[cfg(feature = "wgpu")]
mod provider;
