use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "bitshift",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::bitshift"
)]
pub(crate) async fn bitshift_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BITSHIFT_DESCRIPTOR,
        &runmat_builtins::BITWISE_ERROR_INVALID_INPUT,
        "bitshift",
    )?;
    super::engine::evaluate_bitshift(args).await
}
