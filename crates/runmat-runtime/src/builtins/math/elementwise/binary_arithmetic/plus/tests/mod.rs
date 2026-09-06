use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;

#[cfg(feature = "wgpu")]
fn register_wgpu_provider_available() -> bool {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_ok()
        && runmat_accelerate_api::provider().is_some()
}
use runmat_accelerate_api::HostTensorView;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType, SparseTensor,
    Tensor,
};

const EPS: f64 = 1e-12;

fn plus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::plus_builtin(lhs, rhs, rest))
}

mod host;

mod provider;

#[cfg(feature = "wgpu")]
mod wgpu;
