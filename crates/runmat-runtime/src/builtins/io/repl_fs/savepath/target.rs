use std::path::{Path, PathBuf};

use crate::builtins::common::env as runtime_env;
use crate::builtins::common::fs::{expand_user_path, home_directory};
use runmat_filesystem as vfs;

const DEFAULT_FILENAME: &str = "pathdef.m";

pub(super) struct Target {
    pub(super) path: PathBuf,
    pub(super) create_parent: bool,
}

pub(super) enum ResolveError {
    Status(super::errors::Failure),
    Compatibility(crate::RuntimeError),
}

pub(super) async fn resolve(filename: Option<&str>) -> Result<Target, ResolveError> {
    match filename {
        Some(filename) => explicit(filename).await,
        None => default_target().await,
    }
}

async fn default_target() -> Result<Target, ResolveError> {
    let path = match runtime_env::var("RUNMAT_PATHDEF") {
        Ok(value) if value.trim().is_empty() => {
            return Err(status("savepath: RUNMAT_PATHDEF is empty"));
        }
        Ok(value) => expand(&value).map_err(ResolveError::Status)?,
        Err(_) => home_directory()
            .ok_or_else(|| status("savepath: unable to determine default pathdef location"))?
            .join(".runmat")
            .join(DEFAULT_FILENAME),
    };
    Ok(Target {
        path,
        create_parent: true,
    })
}

async fn explicit(filename: &str) -> Result<Target, ResolveError> {
    let mut path = expand(filename).map_err(ResolveError::Status)?;
    if is_directory_target(&path, filename).await {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::SAVEPATH_DIRECTORY_TARGET_EXTENSION,
            super::errors::NAME,
        )
        .map_err(ResolveError::Compatibility)?;
        path.push(DEFAULT_FILENAME);
    }
    Ok(Target {
        create_parent: crate::compatibility::runmat_extensions_enabled(),
        path,
    })
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

fn status(message: impl Into<String>) -> ResolveError {
    ResolveError::Status(cannot_resolve(message))
}

#[cfg(test)]
pub(super) const fn default_filename() -> &'static str {
    DEFAULT_FILENAME
}
