use runmat_value::Value;

use crate::builtins::table::is_tabular_object;
use crate::BuiltinResult;

use super::error;

pub(super) fn validate(first: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if crate::value_contains_gpu(first) || rest.iter().any(crate::value_contains_gpu) {
        require(&runmat_builtins::FINDGROUPS_RESIDENT_INPUT_EXTENSION)?;
    }
    if is_matrix(first) || rest.iter().any(is_matrix) {
        require(&runmat_builtins::FINDGROUPS_MATRIX_COLUMNS_EXTENSION)?;
    }
    let Value::Object(object) = first else {
        return Ok(());
    };
    if object.is_class(runmat_types::standard::TIMETABLE) {
        require(&runmat_builtins::FINDGROUPS_TIMETABLE_EXTENSION)?;
    }
    if is_tabular_object(object) && !rest.is_empty() {
        require(&runmat_builtins::FINDGROUPS_TABLE_SELECTOR_EXTENSION)?;
        if rest.len() != 1 {
            return Err(error::invalid(
                "findgroups: the table selector form accepts exactly one selector",
            ));
        }
    }
    Ok(())
}

fn require(extension: &runmat_builtins::BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, "findgroups")
}

fn is_matrix(value: &Value) -> bool {
    let shape = match value {
        Value::Tensor(value) => &value.shape,
        Value::LogicalArray(value) => &value.shape,
        Value::StringArray(value) => &value.shape,
        Value::GpuTensor(value) => &value.shape,
        _ => return false,
    };
    shape.first().copied().unwrap_or(1) > 1 && shape.get(1).copied().unwrap_or(1) > 1
}
