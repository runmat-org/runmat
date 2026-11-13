use runmat_value::Value;

use crate::{runtime_descriptor_error, BuiltinResult};

use super::super::working_directory;

const BUILTIN_NAME: &str = "pwd";

pub(super) fn run(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PWD_DESCRIPTOR,
        &runmat_builtins::PWD_ERROR_TOO_MANY_OUTPUTS,
        "pwd",
    )?;
    if !args.is_empty() {
        return Err(runtime_descriptor_error(
            BUILTIN_NAME,
            &runmat_builtins::PWD_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let current = working_directory::query(BUILTIN_NAME, &runmat_builtins::PWD_ERROR_INTERNAL)?;
    Ok(working_directory::value(&current))
}
