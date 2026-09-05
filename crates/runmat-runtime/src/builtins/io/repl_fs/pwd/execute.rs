use runmat_value::Value;

use crate::{runtime_descriptor_error, BuiltinResult};

use super::super::working_directory;

const BUILTIN_NAME: &str = "pwd";

pub(super) fn run(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(runtime_descriptor_error(
            BUILTIN_NAME,
            &runmat_builtins::PWD_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let current = working_directory::query(BUILTIN_NAME, &runmat_builtins::PWD_ERROR_INTERNAL)?;
    Ok(working_directory::value(&current))
}
