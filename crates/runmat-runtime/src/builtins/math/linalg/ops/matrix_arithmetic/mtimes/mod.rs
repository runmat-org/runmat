//! MATLAB-compatible `mtimes` builtin with GPU-aware semantics for RunMat.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

mod errors;
pub(crate) mod spec;
#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const NAME: &str = "mtimes";

#[runtime_builtin(
    name = "mtimes",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes"
)]
async fn mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
    {
        return Err(errors::invalid_input(
            "complex integer arithmetic is not supported",
        ));
    }
    mtimes_eval(&lhs, &rhs).await
}

pub(crate) async fn mtimes_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    super::product::evaluate(lhs, rhs)
        .await
        .map_err(errors::map_control_flow)?
        .map_err(errors::from_product)
}

pub(crate) async fn mtimes_eval_in_context(
    context: &crate::context::RuntimeContext,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<Value> {
    super::product::evaluate_in_context(Some(context), lhs, rhs)
        .await
        .map_err(errors::map_control_flow)?
        .map_err(errors::from_product)
}

#[cfg(test)]
pub(crate) mod tests;
