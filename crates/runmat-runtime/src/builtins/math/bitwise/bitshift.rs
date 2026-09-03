use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitshift",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitshift"
)]
pub(crate) async fn bitshift_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::engine::evaluate_bitshift(args).await
}
