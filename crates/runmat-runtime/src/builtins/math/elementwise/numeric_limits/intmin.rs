use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "intmin",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits::intmin"
)]
pub(super) fn intmin_builtin(arguments: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        runmat_builtins::INTMIN_CATALOG_ENTRY.descriptor,
        &runmat_builtins::NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
        "intmin",
    )?;
    super::execute::run(arguments, super::operation::LimitOperation::INTMIN)
}
