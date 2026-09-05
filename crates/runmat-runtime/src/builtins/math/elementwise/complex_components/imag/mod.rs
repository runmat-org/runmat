//! Imaginary-component extraction.

mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

use super::projection::{self, ProjectionKind};

#[runtime_builtin(
    name = "imag",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::complex_components::imag"
)]
async fn imag_builtin(value: Value) -> BuiltinResult<Value> {
    projection::execute(ProjectionKind::Imaginary, value).await
}

#[cfg(all(test, feature = "wgpu"))]
fn imag_real(value: Value) -> BuiltinResult<Value> {
    futures::executor::block_on(projection::execute(ProjectionKind::Imaginary, value))
}

#[cfg(test)]
async fn imag_gpu(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    projection::execute(ProjectionKind::Imaginary, Value::GpuTensor(handle)).await
}

#[cfg(test)]
mod tests;
