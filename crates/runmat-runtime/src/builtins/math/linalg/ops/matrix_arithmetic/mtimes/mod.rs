//! MATLAB-compatible `mtimes` builtin with GPU-aware semantics for RunMat.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

mod errors;
mod host;
mod integer;
mod provider;
pub(crate) mod spec;

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
    if integer::contains(lhs) || integer::contains(rhs) {
        return integer::evaluate(lhs, rhs).await;
    }
    if let Some(result) = provider::try_product(lhs, rhs).await? {
        return Ok(result);
    }
    host::evaluate(lhs.clone(), rhs.clone()).await
}

#[cfg(test)]
pub(crate) mod tests;
