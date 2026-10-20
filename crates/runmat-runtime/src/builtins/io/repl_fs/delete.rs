//! MATLAB-compatible `delete` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use glob::{Pattern, PatternError};
#[cfg(test)]
use runmat_builtins::DELETE_DESCRIPTOR;
use runmat_builtins::{
    DeleteOption, DELETE_ERROR_EMPTY_FILENAME, DELETE_ERROR_FILE_NOT_FOUND,
    DELETE_ERROR_INVALID_HANDLE, DELETE_ERROR_INVALID_INPUT, DELETE_ERROR_INVALID_PATTERN,
    DELETE_ERROR_IS_DIRECTORY, DELETE_ERROR_OS_ERROR, DELETE_ERROR_RUNTIME_CONTEXT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::fs::{contains_wildcards, expand_user_path, path_to_string};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[path = "delete/error.rs"]
mod error;
use error::*;

#[runtime_builtin(
    name = "delete",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::delete"
)]
async fn delete_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: missing filename input",
        ));
    }
    let mut cooperation = DeleteCooperation::new(context.as_ref());
    if let (1, Some(Value::Pool(pool))) = (args.len(), args.first()) {
        cooperation.completed(0).await?;
        let context = context
            .or_else(crate::context::legacy::active)
            .ok_or_else(|| {
                delete_error_with(
                    &DELETE_ERROR_RUNTIME_CONTEXT,
                    DELETE_ERROR_RUNTIME_CONTEXT.message,
                )
            })?;
        crate::parallel::pool::close(&context, pool)?;
        cooperation.completed(0).await?;
        return Ok(Value::Num(0.0));
    }
    let (operands, resolve_symbolic_links) = parse_delete_options(args.view())?;
    if operands.iter().all(is_handle_input) {
        if resolve_symbolic_links.is_some() {
            return Err(delete_error_with(
                &DELETE_ERROR_INVALID_INPUT,
                "delete: ResolveSymbolicLinks is only valid for filename inputs",
            ));
        }
        let result = cooperation
            .await_io(delete_handles(context.as_ref(), operands))
            .await??;
        cooperation.completed(0).await?;
        return Ok(result);
    }

    if operands.iter().any(contains_handle_input) {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_HANDLE,
            "delete: cannot mix handle and filename inputs",
        ));
    }

    let mut raw_targets = Vec::new();
    for value in operands {
        collect_targets(value, &mut raw_targets)?;
    }

    if raw_targets.is_empty() {
        cooperation.completed(0).await?;
        return Ok(Value::Num(0.0));
    }

    cooperation.completed(0).await?;
    let filesystem = DeleteFilesystem::capture()?;
    for raw in raw_targets {
        delete_target(
            &filesystem,
            &mut cooperation,
            &raw,
            resolve_symbolic_links.unwrap_or(false),
        )
        .await?;
    }
    cooperation.completed(0).await?;
    Ok(Value::Num(0.0))
}

#[path = "delete/cooperation.rs"]
mod cooperation;
use cooperation::DeleteCooperation;

#[path = "delete/input.rs"]
mod input;
use input::*;

#[path = "delete/filesystem.rs"]
mod filesystem;
use filesystem::*;

#[path = "delete/handles.rs"]
mod handles;
use handles::*;

#[cfg(test)]
#[path = "delete/tests.rs"]
pub(crate) mod tests;
