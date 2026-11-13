use runmat_builtins::BinaryBitwiseOperator;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "bitxor",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::binary::bitxor"
)]
pub(crate) async fn bitxor_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BITXOR_DESCRIPTOR,
        &runmat_builtins::BITWISE_ERROR_INVALID_INPUT,
        "bitxor",
    )?;
    super::evaluate(
        "bitxor",
        args,
        BinaryBitwiseOperator::Xor,
        &runmat_builtins::BITXOR_SINGLE_INPUT_EXTENSION,
        &runmat_builtins::BITXOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
        &runmat_builtins::BITXOR_GPU_ASSUMED_TYPE_EXTENSION,
    )
    .await
}
