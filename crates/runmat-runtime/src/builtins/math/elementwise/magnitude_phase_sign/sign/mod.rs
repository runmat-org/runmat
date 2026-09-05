//! MATLAB-compatible `sign` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::SIGN_DESCRIPTOR;
use runmat_builtins::{SIGN_ERROR_INTERNAL, SIGN_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericStorage, Tensor,
    Value,
};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;

use super::errors::MagnitudePhaseSignOperation;
const OPERATION: MagnitudePhaseSignOperation = MagnitudePhaseSignOperation::Sign;
const BUILTIN_NAME: &str = OPERATION.name();

#[runtime_builtin(
    name = "sign",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::magnitude_phase_sign::sign"
)]
async fn sign_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Int(value) => Ok(Value::Int(host::sign_integer_scalar(value))),
        Value::Complex(re, im) => {
            let (re_out, im_out) = host::sign_complex(re, im);
            Ok(Value::Complex(re_out, im_out))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "sign")?;
            host::sign_complex_tensor(ct)
        }
        Value::CharArray(ca) => host::sign_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(OPERATION.error(
            &SIGN_ERROR_INVALID_INPUT,
            "expected numeric, logical, or character input",
        )),
        other => host::sign_real(other),
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
