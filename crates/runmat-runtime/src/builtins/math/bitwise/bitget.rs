use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitget",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitget"
)]
pub(crate) async fn bitget_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BITGET_DESCRIPTOR,
        &runmat_builtins::BITWISE_ERROR_INVALID_INPUT,
        "bitget",
    )?;
    super::engine::evaluate_bitget(args).await
}
