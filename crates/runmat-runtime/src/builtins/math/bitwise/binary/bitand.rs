use runmat_builtins::BinaryBitwiseOperator;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "bitand",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::binary::bitand"
)]
pub(crate) async fn bitand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
