use super::*;
use crate::builtins::common::path_search::{
    find_file_with_extensions, path_is_file, search_directories, GENERAL_FILE_EXTENSIONS,
};
pub(super) async fn resolve_open_path(name: &str) -> BuiltinResult<PathBuf> {
    if let Some(path) = find_exact_open_file(name).await? {
        return Ok(vfs::canonicalize_async(&path).await.unwrap_or(path));
    }

    let path = find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, BUILTIN_NAME)
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        .ok_or_else(|| {
            open_error(
                &OPEN_ERROR_NOT_FOUND,
                format!("open: file or variable was not found: {name}"),
            )
        })?;

    Ok(vfs::canonicalize_async(&path).await.unwrap_or(path))
}

pub(super) async fn find_exact_open_file(name: &str) -> BuiltinResult<Option<PathBuf>> {
    if should_treat_as_path(name) {
        return Ok(None);
    }
    let Some(candidate_name) = exact_filename_candidate(name) else {
        return Ok(None);
    };
    for dir in
        search_directories(BUILTIN_NAME).map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
    {
        let candidate = dir.join(candidate_name);
        if path_is_file(&candidate)
            .await
            .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn exact_filename_candidate(name: &str) -> Option<&str> {
    let path = Path::new(name);
    (path.file_name().is_some() && path.extension().is_some()).then_some(name)
}
