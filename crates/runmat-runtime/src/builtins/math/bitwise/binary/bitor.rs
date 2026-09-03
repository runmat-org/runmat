use runmat_builtins::BinaryBitwiseOperator;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "bitor",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::binary::bitor"
)]
pub(crate) async fn bitor_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::evaluate(
        "bitor",
        args,
        BinaryBitwiseOperator::Or,
        &runmat_builtins::BITOR_SINGLE_INPUT_EXTENSION,
        &runmat_builtins::BITOR_GPU_UNDOCUMENTED_INPUT_EXTENSION,
        &runmat_builtins::BITOR_GPU_ASSUMED_TYPE_EXTENSION,
    )
    .await
}
