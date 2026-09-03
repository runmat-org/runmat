use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitset",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitset"
)]
pub(crate) async fn bitset_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::engine::evaluate_bitset(args).await
}
