use runmat_builtins::FIELDNAMES_OBJECT_FAMILY_EXTENSION;
use runmat_value::Value;

pub(super) fn execute(value: Value) -> crate::BuiltinResult<Value> {
    if matches!(
        &value,
        Value::Object(_) | Value::HandleObject(_) | Value::Listener(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FIELDNAMES_OBJECT_FAMILY_EXTENSION,
            super::BUILTIN_NAME,
        )?;
    }
    super::output::field_name_cell(super::names::collect(&value)?)
}
