use std::path::Path;

use runmat_filesystem as vfs;
use runmat_value::{CharArray, Value};

use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail, BuiltinResult};

const BUILTIN_NAME: &str = "pwd";

pub(super) fn run(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(runtime_descriptor_error(
            BUILTIN_NAME,
            &runmat_builtins::PWD_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let current = vfs::current_dir().map_err(|error| {
        runtime_descriptor_error_with_detail(
            BUILTIN_NAME,
            &runmat_builtins::PWD_ERROR_INTERNAL,
            error.to_string(),
        )
    })?;
    Ok(path_to_value(&current))
}

fn path_to_value(path: &Path) -> Value {
    Value::CharArray(CharArray::new_row(&path.to_string_lossy()))
}
