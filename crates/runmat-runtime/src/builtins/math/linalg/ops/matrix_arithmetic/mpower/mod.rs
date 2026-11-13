//! MATLAB-compatible `mpower` builtin with GPU-aware semantics for RunMat.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

mod errors;
mod evaluate;
mod exponent;
mod provider;
pub(crate) mod spec;

pub(super) const NAME: &str = "mpower";

#[runtime_builtin(
    name = "mpower",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::matrix_arithmetic::mpower"
)]
async fn mpower_builtin(base: Value, exponent: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&base)
        || crate::builtins::common::validation::is_typed_complex_integer(&exponent)
    {
        return Err(errors::invalid_input(
            "complex integer arithmetic is not supported",
        ));
    }
    mpower_eval(&base, &exponent).await
}

pub(crate) async fn mpower_eval(base: &Value, exponent: &Value) -> BuiltinResult<Value> {
    evaluate::evaluate(base, exponent).await
}

#[cfg(test)]
pub(crate) mod tests;
