use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "flintmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits::flintmax"
)]
fn flintmax_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    super::execute::run(arguments, super::operation::LimitOperation::FLINTMAX)
}
