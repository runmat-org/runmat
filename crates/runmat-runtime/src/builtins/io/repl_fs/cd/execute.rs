use runmat_value::Value;

use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail, BuiltinResult};

use super::super::working_directory;

const BUILTIN_NAME: &str = "cd";

pub(super) async fn run(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        0 => current(),
        1 => change(args.into_iter().next().expect("one cd argument")).await,
        _ => Err(runtime_descriptor_error(
            BUILTIN_NAME,
            &runmat_builtins::CD_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

fn current() -> BuiltinResult<Value> {
    let path = working_directory::query(BUILTIN_NAME, &runmat_builtins::CD_ERROR_INTERNAL)?;
    Ok(working_directory::value(&path))
}

async fn change(value: Value) -> BuiltinResult<Value> {
    let (raw, target) = super::input::path(value)?;
    let previous = working_directory::query(BUILTIN_NAME, &runmat_builtins::CD_ERROR_INTERNAL)?;
    runmat_filesystem::set_current_dir_async(&target)
        .await
        .map_err(|error| {
            runtime_descriptor_error_with_detail(
                BUILTIN_NAME,
                &runmat_builtins::CD_ERROR_CHANGE_FAILED,
                format!("to '{raw}' ({error})"),
            )
        })?;
    working_directory::query(BUILTIN_NAME, &runmat_builtins::CD_ERROR_INTERNAL)?;
    Ok(working_directory::value(&previous))
}
