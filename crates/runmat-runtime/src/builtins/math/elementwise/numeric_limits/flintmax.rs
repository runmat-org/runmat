use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "flintmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits::flintmax"
)]
pub(super) fn flintmax_builtin(arguments: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::FLINTMAX_CATALOG_ENTRY.descriptor,
        &runmat_builtins::NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
        "flintmax",
    )?;
    super::execute::run(arguments, super::operation::LimitOperation::FLINTMAX)
}
