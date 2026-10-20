use runmat_builtins::BinaryBitwiseOperator;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "bitand",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::binary::bitand"
)]
pub(crate) async fn bitand_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::BITAND_DESCRIPTOR,
        &runmat_builtins::BITWISE_ERROR_INVALID_INPUT,
        "bitand",
    )?;
    super::evaluate(
        "bitand",
        args,
        BinaryBitwiseOperator::And,
        &runmat_builtins::BITAND_SINGLE_INPUT_EXTENSION,
        &runmat_builtins::BITAND_GPU_UNDOCUMENTED_INPUT_EXTENSION,
        &runmat_builtins::BITAND_GPU_ASSUMED_TYPE_EXTENSION,
    )
    .await
}
