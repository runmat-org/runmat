use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitget",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitget"
)]
pub(crate) async fn bitget_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::engine::evaluate_bitget(args).await
}
