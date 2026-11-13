use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitcmp",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitcmp"
)]
pub(crate) async fn bitcmp_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BITCMP_DESCRIPTOR,
        &runmat_builtins::BITWISE_ERROR_INVALID_INPUT,
        "bitcmp",
    )?;
    super::engine::evaluate_bitcmp(args).await
}
