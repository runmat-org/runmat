use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "intmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits::intmax"
)]
pub(super) fn intmax_builtin(arguments: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::INTMAX_CATALOG_ENTRY.descriptor,
        &runmat_builtins::NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
        "intmax",
    )?;
    super::execute::run(arguments, super::operation::LimitOperation::INTMAX)
}
