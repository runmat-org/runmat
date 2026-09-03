use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitcmp",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitcmp"
)]
pub(crate) async fn bitcmp_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::engine::evaluate_bitcmp(args).await
}
