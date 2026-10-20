use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitset",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitset"
)]
pub(crate) async fn bitset_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BITSET_DESCRIPTOR,
        &runmat_builtins::BITWISE_ERROR_INVALID_INPUT,
        "bitset",
    )?;
    super::engine::evaluate_bitset(args).await
}
