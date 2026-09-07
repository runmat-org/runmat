//! Matrix right-division identity and runtime binding.

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::BuiltinResult;

use super::solve::{self, SolveOrientation};

const NAME: &str = "mrdivide";

pub(crate) mod spec;

#[runtime_builtin(
    name = "mrdivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::ops::matrix_arithmetic::mrdivide"
)]
async fn mrdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    mrdivide_eval(&lhs, &rhs).await
}

pub(crate) async fn mrdivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    solve::evaluate(SolveOrientation::Right, lhs, rhs).await
}

/// Host implementation shared with providers whose solve boundary is CPU-backed.
pub fn mrdivide_host_real_for_provider(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    solve::host_real(SolveOrientation::Right, lhs, rhs)
}

#[cfg(test)]
pub(crate) mod tests;
