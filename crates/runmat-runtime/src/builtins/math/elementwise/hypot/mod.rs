//! MATLAB-compatible `hypot` builtin with GPU-aware semantics for RunMat.

mod admission;
mod errors;
mod host;
mod provider;
mod specification;

#[cfg(target_arch = "wasm32")]
pub(crate) use specification::*;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

const BUILTIN_NAME: &str = "hypot";

#[runtime_builtin(
    name = "hypot",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::hypot"
)]
async fn hypot_builtin(left: Value, right: Value) -> BuiltinResult<Value> {
    admission::output_count()?;
    admission::extensions(&left, &right)?;
    admission::integer_input(&left)?;
    admission::integer_input(&right)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&left, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&right, BUILTIN_NAME)?;
    match (left, right) {
        (Value::GpuTensor(left), Value::GpuTensor(right)) => provider::pair(left, right).await,
        (Value::GpuTensor(left), right) => provider::mixed(left, right, true).await,
        (left, Value::GpuTensor(right)) => provider::mixed(right, left, false).await,
        (left, right) => host::evaluate(left, right),
    }
}

#[cfg(all(test, feature = "wgpu"))]
use host::compute as compute_hypot_tensor;
#[cfg(test)]
use host::{complex_magnitude, scalar as scalar_hypot_value};

#[cfg(test)]
pub(crate) mod tests;
