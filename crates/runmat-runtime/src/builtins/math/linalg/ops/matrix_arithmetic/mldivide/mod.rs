//! Matrix left-division identity and runtime binding.

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::BuiltinResult;

use super::solve::{self, SolveOrientation};

const NAME: &str = "mldivide";

pub(crate) mod spec;

#[runtime_builtin(
    name = "mldivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::matrix_arithmetic::mldivide"
)]
async fn mldivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    mldivide_eval(&lhs, &rhs).await
}

pub(crate) async fn mldivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    solve::evaluate(SolveOrientation::Left, lhs, rhs).await
}

/// Host implementation shared with providers whose solve boundary is CPU-backed.
pub fn mldivide_host_real_for_provider(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    solve::host_real(SolveOrientation::Left, lhs, rhs)
}

#[cfg(test)]
pub(crate) mod tests;
