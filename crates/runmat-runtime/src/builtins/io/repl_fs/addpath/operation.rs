use std::collections::HashSet;
use std::path::Path;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::path_state::current_path_segments;
use crate::builtins::io::repl_fs::path_root::is_rooted_path;
use runmat_filesystem as vfs;

pub(super) async fn plan(request: super::arguments::Request) -> crate::BuiltinResult<String> {
    let mut existing = current_path_segments();
    let mut seen = HashSet::new();
    let mut additions = Vec::new();

    for raw in request.directories {
        let normalized = normalize_directory(&raw).await?;
        let key = super::super::path_list::identity(&normalized);
        if seen.insert(key.clone()) {
            existing.retain(|entry| super::super::path_list::identity(entry) != key);
            additions.push(normalized);
        }
    }

    let segments: Vec<String> = match request.position {
        super::arguments::Position::Begin => additions.into_iter().chain(existing).collect(),
        super::arguments::Position::End => existing.into_iter().chain(additions).collect(),
    };
    Ok(super::super::path_list::join(&segments))
}

async fn normalize_directory(raw: &str) -> crate::BuiltinResult<String> {
    if raw.eq_ignore_ascii_case("pathdef") || raw.eq_ignore_ascii_case("pathdef.m") {
        return Err(super::errors::descriptor(
            &runmat_builtins::ADDPATH_ERROR_PATHDEF,
        ));
    }
    let expanded = expand_user_path(raw, super::errors::NAME).map_err(|error| {
        super::errors::detail(&runmat_builtins::ADDPATH_ERROR_FOLDER_NOT_FOUND, error)
    })?;
    let path = Path::new(&expanded);
    let joined = if is_rooted_path(path) {
        path.to_owned()
    } else {
        vfs::current_dir()
            .map_err(|_| super::errors::descriptor(&runmat_builtins::ADDPATH_ERROR_CURRENT_FOLDER))?
            .join(path)
    };
    let normalized = super::super::path_mutation::lexical::normalize(&joined);
    let metadata = vfs::metadata_async(&normalized).await.map_err(|_| {
        super::errors::detail(&runmat_builtins::ADDPATH_ERROR_FOLDER_NOT_FOUND, raw)
    })?;
    if !metadata.is_dir() {
        return Err(super::errors::detail(
            &runmat_builtins::ADDPATH_ERROR_NOT_FOLDER,
            raw,
        ));
    }
    Ok(path_to_string(&normalized))
}
