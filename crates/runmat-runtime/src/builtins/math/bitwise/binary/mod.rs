pub(crate) mod bitand;
pub(crate) mod bitor;
pub(crate) mod bitxor;

#[cfg(test)]
pub(crate) use bitand::bitand_builtin;
#[cfg(test)]
pub(crate) use bitor::bitor_builtin;
#[cfg(test)]
pub(crate) use bitxor::bitxor_builtin;

use runmat_builtins::{BinaryBitwiseOperator, BuiltinExtensionDescriptor};
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) async fn evaluate(
    name: &'static str,
    args: Vec<Value>,
    operator: BinaryBitwiseOperator,
    single_extension: &BuiltinExtensionDescriptor,
    gpu_domain_extension: &BuiltinExtensionDescriptor,
    gpu_assumed_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<Value> {
    super::engine::evaluate_binary_bitwise(
        name,
        args,
        operator,
        single_extension,
        gpu_domain_extension,
        gpu_assumed_extension,
    )
    .await
}

#[cfg(test)]
mod tests;
