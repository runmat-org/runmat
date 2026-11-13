//! Native-precision conjugate-gradient solve over borrowed dense/CSC matrices
//! and repeatable asynchronous callable operators.

mod async_driver;
mod async_operations;
mod execution;
mod kernel;
mod request;
#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::BuiltinResult;

#[runtime_builtin(
    name = "pcg",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::pcg",
    context = true
)]
async fn pcg_builtin(
    runtime: Option<RuntimeContext>,
    coefficient: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let request = request::parse(coefficient, rhs, rest)?;
    execution::execute(
        request,
        runtime.as_ref(),
        crate::output_count::current_output_count(),
    )
    .await
}
