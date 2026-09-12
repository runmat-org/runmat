use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinErrorDescriptor, MKDIR_ERROR_ABSOLUTE_CHILD, MKDIR_ERROR_ARITY, MKDIR_ERROR_EMPTY_NAME,
    MKDIR_ERROR_FILESYSTEM, MKDIR_ERROR_FOLDER_TYPE, MKDIR_ERROR_NOT_DIRECTORY,
    MKDIR_NOTICE_EXISTS,
};
use runmat_filesystem as vfs;
use runmat_value::Value;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::io::repl_fs::path_root::is_rooted_path;
use crate::{build_runtime_error, BuiltinResult};

use super::super::result::DirectoryOutcome;

pub(super) async fn evaluate(args: Vec<Value>) -> BuiltinResult<DirectoryOutcome> {
    let args = super::super::input::gather(args, "mkdir").await?;
    let target = match args.as_slice() {
        [folder] => single_target(folder)?,
        [parent, child] => child_target(parent, child)?,
        _ => return Err(error(&MKDIR_ERROR_ARITY)),
    };
    Ok(create(&target).await)
}

fn single_target(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = folder_text(value)?;
    nonempty(&raw)?;
    expand(&raw)
}

fn child_target(parent: &Value, child: &Value) -> BuiltinResult<PathBuf> {
    let parent = folder_text(parent)?;
    let child = folder_text(child)?;
    nonempty(&parent)?;
    nonempty(&child)?;
    let child = PathBuf::from(child);
    if is_rooted_path(&child) {
        return Err(error(&MKDIR_ERROR_ABSOLUTE_CHILD));
    }
    Ok(expand(&parent)?.join(child))
}

fn folder_text(value: &Value) -> BuiltinResult<String> {
    super::super::input::text_scalar(value).map_err(|_| error(&MKDIR_ERROR_FOLDER_TYPE))
}

fn nonempty(value: &str) -> BuiltinResult<()> {
    if value.is_empty() {
        Err(error(&MKDIR_ERROR_EMPTY_NAME))
    } else {
        Ok(())
    }
}

fn expand(value: &str) -> BuiltinResult<PathBuf> {
    expand_user_path(value, "mkdir")
        .map(PathBuf::from)
        .map_err(|cause| error_with_source(&MKDIR_ERROR_FILESYSTEM, cause.into()))
}

async fn create(path: &Path) -> DirectoryOutcome {
    match vfs::metadata_async(path).await {
        Ok(metadata) if metadata.is_dir() => DirectoryOutcome::notice(
            MKDIR_NOTICE_EXISTS.message,
            identifier(&MKDIR_NOTICE_EXISTS),
        ),
        Ok(_) => DirectoryOutcome::failure(
            format!(
                "Cannot create folder \"{}\": the path is not a directory.",
                path.display()
            ),
            identifier(&MKDIR_ERROR_NOT_DIRECTORY),
        ),
        Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => {
            match vfs::create_dir_all_async(path).await {
                Ok(()) => DirectoryOutcome::Success,
                Err(cause) => filesystem_failure(path, cause),
            }
        }
        Err(cause) => filesystem_failure(path, cause),
    }
}

fn filesystem_failure(path: &Path, cause: std::io::Error) -> DirectoryOutcome {
    DirectoryOutcome::failure(
        format!("Unable to create folder \"{}\": {cause}", path.display()),
        identifier(&MKDIR_ERROR_FILESYSTEM),
    )
}

fn error(descriptor: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    let mut builder = build_runtime_error(descriptor.message).with_builtin("mkdir");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn error_with_source(
    descriptor: &'static BuiltinErrorDescriptor,
    cause: crate::RuntimeError,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(descriptor.message)
        .with_builtin("mkdir")
        .with_source(cause);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn identifier(descriptor: &'static BuiltinErrorDescriptor) -> &'static str {
    descriptor.identifier.unwrap_or("RunMat:mkdir:Error")
}
