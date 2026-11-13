use super::super::record::Record;
use super::authority::DirFilesystem;
use super::directory::{absolute_folder, directory};
use crate::builtins::common::fs::{contains_wildcards, expand_user_path};
use crate::BuiltinResult;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub(super) async fn from_text(fs: &DirFilesystem, text: &str) -> BuiltinResult<Vec<Record>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return directory(fs, fs.current_directory(), true).await;
    }
    let expanded = expand_user_path(trimmed, "dir").map_err(super::super::error::operation)?;
    if contains_wildcards(&expanded) {
        pattern(fs, Path::new(&expanded), trimmed).await
    } else {
        path(fs, Path::new(&expanded), trimmed).await
    }
}

pub(super) async fn folder_pattern(
    fs: &DirFilesystem,
    folder: &str,
    pattern_text: &str,
) -> BuiltinResult<Vec<Record>> {
    let expanded =
        expand_user_path(folder.trim(), "dir").map_err(super::super::error::operation)?;
    if contains_wildcards(&expanded) {
        return Err(super::super::error::contract(
            &runmat_builtins::DIR_ERROR_FOLDER_WILDCARD,
        ));
    }
    let folder = PathBuf::from(expanded);
    let pattern_text = pattern_text.trim();
    if pattern_text.is_empty() {
        return directory(fs, &folder, true).await;
    }
    let target = folder.join(pattern_text);
    if contains_wildcards(pattern_text) {
        pattern(fs, &target, pattern_text).await
    } else {
        path(fs, &target, pattern_text).await
    }
}

async fn path(fs: &DirFilesystem, path: &Path, display: &str) -> BuiltinResult<Vec<Record>> {
    match fs.metadata(path).await {
        Ok(metadata) if metadata.is_dir() => directory(fs, path, true).await,
        Ok(metadata) => {
            let folder_path = path.parent().unwrap_or_else(|| Path::new("."));
            let folder = absolute_folder(fs, folder_path).await?;
            let name = crate::builtins::common::fs::enumeration::filename(path)
                .to_string_lossy()
                .into_owned();
            Ok(vec![Record::from_metadata(folder, name, &metadata)])
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(super::super::error::io(
            format!("dir: unable to access '{display}' ({error})"),
            error,
        )),
    }
}

async fn pattern(fs: &DirFilesystem, pattern: &Path, display: &str) -> BuiltinResult<Vec<Record>> {
    let entries = fs.matching(pattern).await.map_err(|error| {
        super::super::error::io(
            format!("dir: unable to enumerate matches for '{display}' ({error})"),
            error,
        )
    })?;
    let mut records = Vec::with_capacity(entries.len());
    for entry in entries {
        let folder =
            absolute_folder(fs, entry.path.parent().unwrap_or_else(|| Path::new("."))).await?;
        let name = crate::builtins::common::fs::enumeration::filename(&entry.path)
            .to_string_lossy()
            .into_owned();
        records.push(Record::from_metadata(folder, name, &entry.metadata));
    }
    Ok(records)
}
