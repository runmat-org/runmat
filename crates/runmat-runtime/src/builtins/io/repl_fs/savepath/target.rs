use std::path::{Path, PathBuf};

use crate::builtins::common::env as runtime_env;
use crate::builtins::common::fs::{expand_user_path, home_directory};
use runmat_filesystem as vfs;

const DEFAULT_FILENAME: &str = "pathdef.m";

pub(super) struct Target {
    pub(super) path: PathBuf,
    pub(super) create_parent: bool,
}

// Operational errors use the canonical outer channel without allocating a
// second envelope. Domain failures remain savepath's status-return outcome.
pub(super) async fn resolve(
    filename: Option<&str>,
) -> crate::BuiltinResult<Result<Target, super::errors::Failure>> {
    match filename {
        Some(filename) => explicit(filename).await,
        None => Ok(default_target()),
    }
}

fn default_target() -> Result<Target, super::errors::Failure> {
    let path = match runtime_env::var("RUNMAT_PATHDEF") {
        Ok(value) if value.trim().is_empty() => {
            return Err(cannot_resolve("savepath: RUNMAT_PATHDEF is empty"));
        }
        Ok(value) => expand(&value)?,
        Err(_) => home_directory()
            .ok_or_else(|| {
                cannot_resolve("savepath: unable to determine default pathdef location")
            })?
            .join(".runmat")
            .join(DEFAULT_FILENAME),
    };
    Ok(Target {
        path,
        create_parent: true,
    })
}

async fn explicit(filename: &str) -> crate::BuiltinResult<Result<Target, super::errors::Failure>> {
    let mut path = match expand(filename) {
        Ok(path) => path,
        Err(failure) => return Ok(Err(failure)),
    };
    if is_directory_target(&path, filename).await {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::SAVEPATH_DIRECTORY_TARGET_EXTENSION,
            super::errors::NAME,
        )?;
        path.push(DEFAULT_FILENAME);
    }
    Ok(Ok(Target {
        create_parent: crate::compatibility::runmat_extensions_enabled(),
        path,
    }))
}

fn expand(filename: &str) -> Result<PathBuf, super::errors::Failure> {
    expand_user_path(filename, super::errors::NAME)
        .map(PathBuf::from)
        .map_err(cannot_resolve)
}

async fn is_directory_target(path: &Path, original: &str) -> bool {
    if has_directory_suffix(original) {
        return true;
    }
    vfs::metadata_async(path)
        .await
        .is_ok_and(|metadata| metadata.is_dir())
}

fn has_directory_suffix(path: &str) -> bool {
    path.ends_with(std::path::MAIN_SEPARATOR)
        || path.ends_with('/')
        || cfg!(windows) && path.ends_with('\\')
}

fn cannot_resolve(message: impl Into<String>) -> super::errors::Failure {
    super::errors::Failure::new(&runmat_builtins::SAVEPATH_ERROR_CANNOT_RESOLVE, message)
}

#[cfg(test)]
pub(super) const fn default_filename() -> &'static str {
    DEFAULT_FILENAME
}
