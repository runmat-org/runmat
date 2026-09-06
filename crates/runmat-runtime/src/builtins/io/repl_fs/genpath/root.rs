use std::path::{Path, PathBuf};

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use runmat_filesystem as vfs;

pub(super) struct Root {
    pub(super) path: PathBuf,
    pub(super) canonical: String,
}

pub(super) async fn resolve(text: Option<&str>) -> crate::BuiltinResult<Root> {
    match text {
        Some(text) => resolve_named(text).await,
        None => {
            let cwd = vfs::current_dir().map_err(|error| {
                super::errors::detail(
                    &runmat_builtins::GENPATH_ERROR_CURRENT_FOLDER,
                    error.to_string(),
                )
            })?;
            resolve_path(cwd, "current directory").await
        }
    }
}

async fn resolve_named(text: &str) -> crate::BuiltinResult<Root> {
    if text.trim().is_empty() {
        return Err(super::errors::detail(
            &runmat_builtins::GENPATH_ERROR_FOLDER_NOT_FOUND,
            text,
        ));
    }
    let expanded = expand_user_path(text, super::errors::NAME).map_err(|error| {
        super::errors::detail(&runmat_builtins::GENPATH_ERROR_FOLDER_NOT_FOUND, error)
    })?;
    let path = PathBuf::from(expanded);
    let absolute = if super::super::is_rooted_path(&path) {
        path
    } else {
        let cwd = vfs::current_dir().map_err(|error| {
            super::errors::detail(
                &runmat_builtins::GENPATH_ERROR_CURRENT_FOLDER,
                error.to_string(),
            )
        })?;
        cwd.join(path)
    };
    resolve_path(absolute, text).await
}

async fn resolve_path(path: PathBuf, display: &str) -> crate::BuiltinResult<Root> {
    let canonical = vfs::canonicalize_async(&path).await.map_err(|_| {
        super::errors::detail(&runmat_builtins::GENPATH_ERROR_FOLDER_NOT_FOUND, display)
    })?;
    let metadata = vfs::metadata_async(&canonical).await.map_err(|_| {
        super::errors::detail(&runmat_builtins::GENPATH_ERROR_FOLDER_NOT_FOUND, display)
    })?;
    if !metadata.is_dir() {
        return Err(super::errors::detail(
            &runmat_builtins::GENPATH_ERROR_NOT_FOLDER,
            display,
        ));
    }
    Ok(Root {
        canonical: canonical_string(&canonical),
        path: canonical,
    })
}

#[cfg(windows)]
pub(super) fn canonical_string(path: &Path) -> String {
    path_to_string(path)
        .strip_prefix(r"\\?\")
        .map_or_else(|| path_to_string(path), str::to_owned)
}

#[cfg(not(windows))]
pub(super) fn canonical_string(path: &Path) -> String {
    path_to_string(path)
}

#[cfg(test)]
pub(super) fn canonical(path: &Path) -> String {
    futures::executor::block_on(resolve_path(path.to_path_buf(), &path_to_string(path)))
        .expect("canonical folder")
        .canonical
}
