//! MATLAB-compatible `abs` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ABS_ERROR_INTERNAL, ABS_ERROR_INVALID_INPUT, ABS_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericStorage,
    SparseTensor, Tensor, Value,
};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::symbolic::symbolic_expr_to_value;
use crate::BuiltinResult;

use super::errors::MagnitudePhaseSignOperation;
const OPERATION: MagnitudePhaseSignOperation = MagnitudePhaseSignOperation::Magnitude;
const BUILTIN_NAME: &str = OPERATION.name();

#[runtime_builtin(
    name = "abs",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::magnitude_phase_sign::abs"
)]
async fn abs_builtin(value: Value) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(OPERATION.error(&ABS_ERROR_TOO_MANY_OUTPUTS, "expected at most one output"));
    }
    extensions::ensure(&value)?;
    match value {
        Value::Symbolic(expr) => Ok(symbolic_expr_to_value(
            runmat_value::SymbolicExpr::function_call("abs", vec![expr]),
        )),
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Int(value) => Ok(Value::Int(host::abs_integer_scalar(value))),
        Value::Complex(re, im) => Ok(Value::Num(host::complex_magnitude(re, im))),
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "abs")?;
            host::abs_complex_tensor(ct)
        }
        Value::SparseTensor(sparse) => host::abs_sparse_tensor(sparse),
        Value::CharArray(ca) => host::abs_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(OPERATION.error(&ABS_ERROR_INVALID_INPUT, "expected numeric input"))
        }
        other => host::abs_real(other),
    }
}

mod extensions;
mod host;
mod provider;
mod specs;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[cfg(test)]
mod tests;
