//! Real-component extraction.

mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

use super::projection::{self, ProjectionKind};

#[runtime_builtin(
    name = "real",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::complex_components::real"
)]
async fn real_builtin(value: Value) -> BuiltinResult<Value> {
    projection::execute(ProjectionKind::Real, value).await
}

#[cfg(all(test, feature = "wgpu"))]
fn real_real(value: Value) -> BuiltinResult<Value> {
    futures::executor::block_on(projection::execute(ProjectionKind::Real, value))
}

#[cfg(all(test, feature = "wgpu"))]
async fn real_gpu(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    projection::execute(ProjectionKind::Real, Value::GpuTensor(handle)).await
}

#[cfg(test)]
mod tests;
