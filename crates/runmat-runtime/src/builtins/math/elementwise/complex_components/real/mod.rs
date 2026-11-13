//! Real-component extraction.

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
    name = "real",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::complex_components::real"
)]
async fn real_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::REAL_DESCRIPTOR,
        &runmat_builtins::REAL_ERROR_TOO_MANY_OUTPUTS,
        "real",
    )?;
    projection::execute(runtime.as_ref(), ProjectionKind::Real, value).await
}

#[cfg(all(test, feature = "wgpu"))]
fn real_real(value: Value) -> BuiltinResult<Value> {
    futures::executor::block_on(projection::execute(None, ProjectionKind::Real, value))
        .map(|owned| owned.value().clone())
}

#[cfg(all(test, feature = "wgpu"))]
async fn real_gpu(handle: runmat_accelerate_api::GpuTensorHandle) -> BuiltinResult<Value> {
    projection::execute(None, ProjectionKind::Real, Value::GpuTensor(handle))
        .await
        .map(|owned| owned.value().clone())
}

#[cfg(test)]
mod tests;
