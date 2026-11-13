//! Imaginary-component extraction.

mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;

use super::projection::{self, ProjectionKind};

#[runtime_builtin(
    name = "imag",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::complex_components::imag"
)]
async fn imag_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::IMAG_DESCRIPTOR,
        &runmat_builtins::IMAG_ERROR_TOO_MANY_OUTPUTS,
        "imag",
    )?;
    projection::execute(runtime.as_ref(), ProjectionKind::Imaginary, value).await
}

#[cfg(all(test, feature = "wgpu"))]
fn imag_real(value: Value) -> BuiltinResult<Value> {
    futures::executor::block_on(projection::execute(None, ProjectionKind::Imaginary, value))
        .map(|owned| owned.value().clone())
}

#[cfg(test)]
async fn imag_gpu(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    projection::execute(None, ProjectionKind::Imaginary, Value::GpuTensor(handle))
        .await
        .map(|owned| owned.value().clone())
}

#[cfg(test)]
mod tests;
