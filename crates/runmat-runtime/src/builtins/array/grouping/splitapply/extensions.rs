use runmat_value::Value;

use crate::BuiltinResult;

pub(super) fn validate(group_numbers: &Value) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(group_numbers) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::SPLITAPPLY_INTEGER_GROUP_EXTENSION,
            "splitapply",
        )?;
    }
    Ok(())
}
