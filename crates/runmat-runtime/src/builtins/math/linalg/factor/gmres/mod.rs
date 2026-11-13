//! Restarted GMRES over borrowed dense/CSC and repeatable callable operators.

mod execution;
mod output;
mod recurrence;
mod request;
mod scalar;
mod workspace;
#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::BuiltinResult;

#[runtime_builtin(
    name = "gmres",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::gmres",
    context = true
)]
async fn gmres_builtin(
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
