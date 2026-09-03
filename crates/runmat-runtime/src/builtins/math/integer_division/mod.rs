use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

mod engine;

#[runtime_builtin(
    name = "idivide",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::integer_division"
)]
pub(crate) async fn idivide_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    engine::evaluate(args).await
}

#[cfg(test)]
mod tests;
