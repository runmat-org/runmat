//! MATLAB-compatible `angle` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::{ANGLE_DESCRIPTOR, ANGLE_INTEGER_CAPABILITIES};
use runmat_builtins::{ANGLE_ERROR_INTERNAL, ANGLE_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;

use super::errors::MagnitudePhaseSignOperation;
const OPERATION: MagnitudePhaseSignOperation = MagnitudePhaseSignOperation::Phase;
const BUILTIN_NAME: &str = OPERATION.name();

#[runtime_builtin(
    name = "angle",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::magnitude_phase_sign::angle"
)]
async fn angle_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Complex(re, im) => Ok(Value::Num(host::angle_scalar(re, im))),
        Value::ComplexTensor(ct) => {
            if ct.integer_storage().is_some() {
                return Err(OPERATION.error(
                    &ANGLE_ERROR_INVALID_INPUT,
                    "expected single or double input",
                ));
            }
            host::angle_complex_tensor(ct)
        }
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => Err(OPERATION.error(
            &ANGLE_ERROR_INVALID_INPUT,
            "expected single or double input",
        )),
        other => host::angle_real(other),
    }
}

mod host;
mod provider;
mod specs;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[cfg(test)]
mod tests;
