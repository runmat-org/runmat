use std::path::Path;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{CharArray, Value};

use crate::{runtime_descriptor_error_with_detail, BuiltinResult};

pub(super) fn query(
    builtin: &'static str,
    unavailable: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<std::path::PathBuf> {
    runmat_filesystem::current_dir().map_err(|error| {
        runtime_descriptor_error_with_detail(builtin, unavailable, error.to_string())
    })
}

pub(super) fn value(path: &Path) -> Value {
    Value::CharArray(CharArray::new_row(&path.to_string_lossy()))
}
