use std::collections::HashSet;
use std::path::Path;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::path_state::current_path_segments;
use crate::builtins::io::repl_fs::path_root::is_rooted_path;
use runmat_filesystem as vfs;

pub(super) async fn plan(directories: Vec<String>) -> crate::BuiltinResult<String> {
    let mut segments = current_path_segments();
    let mut seen = HashSet::new();
    for raw in directories {
        let key = super::super::path_list::identity(&raw);
        if seen.insert(key) {
            remove(&mut segments, &raw).await?;
        }
    }
    Ok(super::super::path_list::join(&segments))
}

async fn remove(segments: &mut Vec<String>, raw: &str) -> crate::BuiltinResult<()> {
    if retain_other_entries(segments, raw) {
        return Ok(());
    }
    let normalized = normalize(raw)?;
    if retain_other_entries(segments, &path_to_string(&normalized)) {
        return Ok(());
    }
    let metadata = vfs::metadata_async(&normalized)
        .await
        .map_err(|_| super::errors::detail(&runmat_builtins::RMPATH_ERROR_FOLDER_NOT_FOUND, raw))?;
    if metadata.is_dir() {
        Err(super::errors::detail(
            &runmat_builtins::RMPATH_ERROR_NOT_ON_PATH,
            raw,
        ))
    } else {
        Err(super::errors::detail(
            &runmat_builtins::RMPATH_ERROR_NOT_FOLDER,
            raw,
        ))
    }
}

fn retain_other_entries(segments: &mut Vec<String>, requested: &str) -> bool {
    let requested = super::super::path_list::identity(requested);
    let before = segments.len();
    segments.retain(|entry| super::super::path_list::identity(entry) != requested);
    segments.len() != before
}

fn normalize(raw: &str) -> crate::BuiltinResult<std::path::PathBuf> {
    let expanded = expand_user_path(raw, super::errors::NAME).map_err(|error| {
        super::errors::detail(&runmat_builtins::RMPATH_ERROR_FOLDER_NOT_FOUND, error)
    })?;
    let path = Path::new(&expanded);
    let joined = if is_rooted_path(path) {
        path.to_owned()
    } else {
        vfs::current_dir()
            .map_err(|_| super::errors::descriptor(&runmat_builtins::RMPATH_ERROR_CURRENT_FOLDER))?
            .join(path)
    };
    Ok(super::super::path_mutation::lexical::normalize(&joined))
}
