use super::*;

pub(super) async fn delete_target(raw: &str, resolve_symbolic_links: bool) -> BuiltinResult<()> {
    let expanded = expand_user_path(raw, "delete")
        .map_err(|msg| delete_error_with(&DELETE_ERROR_INVALID_INPUT, msg))?;
    if expanded.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_EMPTY_FILENAME,
            DELETE_ERROR_EMPTY_FILENAME.message,
        ));
    }

    if contains_wildcards(&expanded) {
        delete_with_pattern(&expanded, raw, resolve_symbolic_links).await
    } else {
        delete_single_path_async(&PathBuf::from(&expanded), raw, resolve_symbolic_links).await
    }
}

async fn delete_with_pattern(
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

    let paths = match glob::glob(pattern) {
        Ok(iter) => iter,
        Err(PatternError { msg, .. }) => {
            return Err(delete_error_with(
                &DELETE_ERROR_INVALID_PATTERN,
                format!("delete: invalid wildcard pattern '{display}' ({msg})"),
            ))
        }
    };

    let mut matches = Vec::new();
    for entry in paths {
        match entry {
            Ok(path) => matches.push(path),
            Err(err) => {
                let problem_path = path_to_string(err.path());
                return Err(delete_error_with(
                    &DELETE_ERROR_OS_ERROR,
                    format!(
                        "delete: unable to delete '{}' ({})",
                        problem_path,
                        err.error()
                    ),
                ));
            }
        }
    }

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
        delete_single_path_async(&path, &display_path, resolve_symbolic_links).await?;
    }
    Ok(())
}

async fn delete_single_path_async(
    path: &Path,
    display: &str,
    resolve_symbolic_links: bool,
) -> BuiltinResult<()> {
    let link_metadata = vfs::symlink_metadata_async(path).await;
    if resolve_symbolic_links
        && link_metadata
            .as_ref()
            .is_ok_and(runmat_filesystem::FsMetadata::is_symlink)
    {
        let target = vfs::canonicalize_async(path).await.map_err(|err| {
            delete_error_with(
                &DELETE_ERROR_OS_ERROR,
                format!(
                    "delete: unable to resolve symbolic link '{}' ({err})",
                    display
                ),
            )
        })?;
        let metadata = vfs::metadata_async(&target).await.map_err(|err| {
            delete_error_with(
                &DELETE_ERROR_OS_ERROR,
                format!(
                    "delete: unable to inspect symbolic-link target '{}' ({err})",
                    display
                ),
            )
        })?;
        return if metadata.is_dir() {
            vfs::remove_dir_async(&target).await
        } else {
            vfs::remove_file_async(&target).await
        }
        .map_err(|err| {
            delete_error_with(
                &DELETE_ERROR_OS_ERROR,
                format!(
                    "delete: unable to delete symbolic-link target '{}' ({err})",
                    display
                ),
            )
        });
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
            vfs::remove_file_async(path).await.map_err(|err| {
                delete_error_with(
                    &DELETE_ERROR_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                )
            })
        }
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                Err(delete_error_with(
                    &DELETE_ERROR_FILE_NOT_FOUND,
                    format!(
                        "delete: cannot delete '{}' because it does not exist",
                        display
                    ),
                ))
            } else {
                Err(delete_error_with(
                    &DELETE_ERROR_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                ))
            }
        }
    }
}

#[cfg(test)]
pub(super) fn delete_single_path(path: &Path, display: &str) -> BuiltinResult<()> {
    futures::executor::block_on(delete_single_path_async(path, display, false))
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
