use super::*;
use std::sync::Arc;

pub(super) struct DeleteFilesystem {
    provider: Arc<dyn vfs::FsProvider>,
    cwd: PathBuf,
    resource_lease: Option<vfs::CapturedFsResourceLease>,
}

impl DeleteFilesystem {
    pub(super) fn capture() -> BuiltinResult<Self> {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|error| {
            delete_io_error(
                &DELETE_ERROR_OS_ERROR,
                format!("delete: unable to determine current directory ({error})"),
                error,
            )
        })?;
        let resource_lease = vfs::capture_resource_lease(provider.clone());
        Ok(Self {
            provider,
            cwd,
            resource_lease,
        })
    }

    fn resolved(&self, path: &Path) -> PathBuf {
        if self.provider.path_namespace().is_absolute(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    fn checkpoint(&self) -> BuiltinResult<()> {
        if self
            .resource_lease
            .as_ref()
            .is_some_and(|lease| !lease.is_valid())
        {
            Err(delete_error_with(
                &DELETE_ERROR_OS_ERROR,
                "delete: filesystem resource authority changed during operation",
            ))
        } else {
            Ok(())
        }
    }
}

pub(super) async fn delete_target(
    fs: &DeleteFilesystem,
    cooperation: &mut DeleteCooperation,
    raw: &str,
    resolve_symbolic_links: bool,
) -> BuiltinResult<()> {
    let expanded = expand_user_path(raw, "delete")
        .map_err(|msg| delete_error_with(&DELETE_ERROR_INVALID_INPUT, msg))?;
    if expanded.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_EMPTY_FILENAME,
            DELETE_ERROR_EMPTY_FILENAME.message,
        ));
    }

    if contains_wildcards(&expanded) {
        delete_with_pattern(fs, cooperation, &expanded, raw, resolve_symbolic_links).await
    } else {
        delete_single_path_async(
            fs,
            cooperation,
            &PathBuf::from(&expanded),
            raw,
            resolve_symbolic_links,
        )
        .await
    }
}

async fn delete_with_pattern(
    fs: &DeleteFilesystem,
    cooperation: &mut DeleteCooperation,
    pattern: &str,
    display: &str,
    resolve_symbolic_links: bool,
) -> BuiltinResult<()> {
    validate_wildcard_pattern(pattern, display)?;

    if let Err(PatternError { msg, .. }) = Pattern::new(pattern) {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_PATTERN,
            format!("delete: invalid wildcard pattern '{display}' ({msg})"),
        ));
    }

    fs.checkpoint()?;
    let mut matches: Vec<PathBuf> = cooperation
        .await_io(crate::builtins::common::fs::enumeration::matching_in(
            fs.provider.as_ref(),
            &fs.cwd,
            Path::new(pattern),
        ))
        .await?
        .map_err(|error| {
            delete_io_error(
                &DELETE_ERROR_OS_ERROR,
                format!("delete: unable to enumerate matches for '{display}' ({error})"),
                error,
            )
        })?
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    fs.checkpoint()?;
    matches.sort();

    if matches.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_FILE_NOT_FOUND,
            format!(
                "delete: cannot delete '{}' because it does not exist",
                display
            ),
        ));
    }

    for path in matches {
        let display_path = path_to_string(&path);
        delete_single_path_async(
            fs,
            cooperation,
            &path,
            &display_path,
            resolve_symbolic_links,
        )
        .await?;
    }
    Ok(())
}

async fn delete_single_path_async(
    fs: &DeleteFilesystem,
    cooperation: &mut DeleteCooperation,
    path: &Path,
    display: &str,
    resolve_symbolic_links: bool,
) -> BuiltinResult<()> {
    let path = fs.resolved(path);
    fs.checkpoint()?;
    let link_metadata = cooperation
        .await_io(fs.provider.symlink_metadata(&path))
        .await?;
    fs.checkpoint()?;
    if resolve_symbolic_links
        && link_metadata
            .as_ref()
            .is_ok_and(runmat_filesystem::FsMetadata::is_symlink)
    {
        let target = cooperation
            .await_io(fs.provider.canonicalize(&path))
            .await?
            .map_err(|err| {
                delete_io_error(
                    &DELETE_ERROR_OS_ERROR,
                    format!(
                        "delete: unable to resolve symbolic link '{}' ({err})",
                        display
                    ),
                    err,
                )
            })?;
        fs.checkpoint()?;
        let metadata = cooperation
            .await_io(fs.provider.metadata(&target))
            .await?
            .map_err(|err| {
                delete_io_error(
                    &DELETE_ERROR_OS_ERROR,
                    format!(
                        "delete: unable to inspect symbolic-link target '{}' ({err})",
                        display
                    ),
                    err,
                )
            })?;
        fs.checkpoint()?;
        let removal = if metadata.is_dir() {
            cooperation
                .await_io(fs.provider.remove_dir(&target))
                .await?
        } else {
            cooperation
                .await_io(fs.provider.remove_file(&target))
                .await?
        };
        removal.map_err(|err| {
            delete_io_error(
                &DELETE_ERROR_OS_ERROR,
                format!(
                    "delete: unable to delete symbolic-link target '{}' ({err})",
                    display
                ),
                err,
            )
        })?;
        return fs.checkpoint();
    }
    match link_metadata {
        Ok(meta) => {
            if meta.is_dir() && !meta.is_symlink() {
                return Err(delete_error_with(
                    &DELETE_ERROR_IS_DIRECTORY,
                    format!(
                        "delete: cannot delete '{}' because it is a directory (use rmdir instead)",
                        display
                    ),
                ));
            }
            fs.checkpoint()?;
            cooperation
                .await_io(fs.provider.remove_file(&path))
                .await?
                .map_err(|err| {
                    delete_io_error(
                        &DELETE_ERROR_OS_ERROR,
                        format!("delete: unable to delete '{}' ({})", display, err),
                        err,
                    )
                })?;
            fs.checkpoint()
        }
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                Err(delete_io_error(
                    &DELETE_ERROR_FILE_NOT_FOUND,
                    format!(
                        "delete: cannot delete '{}' because it does not exist",
                        display
                    ),
                    err,
                ))
            } else {
                Err(delete_io_error(
                    &DELETE_ERROR_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                    err,
                ))
            }
        }
    }
}

#[cfg(test)]
pub(super) fn delete_single_path(path: &Path, display: &str) -> BuiltinResult<()> {
    let fs = DeleteFilesystem::capture()?;
    let mut cooperation = DeleteCooperation::new(None);
    futures::executor::block_on(delete_single_path_async(
        &fs,
        &mut cooperation,
        path,
        display,
        false,
    ))
}

fn validate_wildcard_pattern(pattern: &str, display: &str) -> BuiltinResult<()> {
    if has_unbalanced(pattern, '[', ']') || has_unbalanced(pattern, '{', '}') {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_PATTERN,
            format!("delete: invalid wildcard pattern '{display}'"),
        ));
    }
    Ok(())
}

fn has_unbalanced(pattern: &str, open: char, close: char) -> bool {
    let mut depth = 0usize;
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // Skip escaped characters to avoid false positives
            let _ = chars.next();
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            if depth == 0 {
                return true;
            }
            depth -= 1;
        }
    }
    depth != 0
}
