use runmat_filesystem as vfs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::builtins::common::fs::{
    compare_names, contains_wildcards, expand_user_path, path_to_string,
};
use crate::BuiltinResult;

use super::input::Input;
use super::record::Record;

pub(super) async fn execute(input: Input) -> BuiltinResult<Vec<Record>> {
    let mut records = match input {
        Input::Current => directory(&current_directory()?, true).await?,
        Input::Name(name) => from_text(&name).await?,
        Input::FolderPattern { folder, pattern } => folder_pattern(&folder, &pattern).await?,
    };
    records.sort_by(|left, right| compare_names(&left.name, &right.name));
    Ok(records)
}

async fn from_text(text: &str) -> BuiltinResult<Vec<Record>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return directory(&current_directory()?, true).await;
    }
    let expanded = expand_user_path(trimmed, "dir").map_err(super::error::operation)?;
    if contains_wildcards(&expanded) {
        pattern(Path::new(&expanded), trimmed).await
    } else {
        path(Path::new(&expanded), trimmed).await
    }
}

async fn folder_pattern(folder: &str, pattern_text: &str) -> BuiltinResult<Vec<Record>> {
    let expanded = expand_user_path(folder.trim(), "dir").map_err(super::error::operation)?;
    if contains_wildcards(&expanded) {
        return Err(super::error::contract(
            &runmat_builtins::DIR_ERROR_FOLDER_WILDCARD,
        ));
    }
    let folder = PathBuf::from(expanded);
    let pattern_text = pattern_text.trim();
    if pattern_text.is_empty() {
        return directory(&folder, true).await;
    }
    let target = folder.join(pattern_text);
    if contains_wildcards(pattern_text) {
        pattern(&target, pattern_text).await
    } else {
        path(&target, pattern_text).await
    }
}

async fn path(path: &Path, display: &str) -> BuiltinResult<Vec<Record>> {
    match vfs::metadata_async(path).await {
        Ok(metadata) if metadata.is_dir() => directory(path, true).await,
        Ok(metadata) => {
            let folder_path = path.parent().unwrap_or_else(|| Path::new("."));
            let folder = absolute_folder(folder_path).await?;
            let name = super::super::enumeration::filename(path)
                .to_string_lossy()
                .into_owned();
            Ok(vec![Record::from_metadata(folder, name, &metadata)])
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(super::error::operation(format!(
            "dir: unable to access '{display}' ({error})"
        ))),
    }
}

async fn pattern(pattern: &Path, display: &str) -> BuiltinResult<Vec<Record>> {
    let entries = super::super::enumeration::matching(pattern)
        .await
        .map_err(|error| {
            super::error::operation(format!(
                "dir: unable to enumerate matches for '{display}' ({error})"
            ))
        })?;
    let mut records = Vec::with_capacity(entries.len());
    for entry in entries {
        let folder = absolute_folder(entry.path.parent().unwrap_or_else(|| Path::new("."))).await?;
        let name = super::super::enumeration::filename(&entry.path)
            .to_string_lossy()
            .into_owned();
        records.push(Record::from_metadata(folder, name, &entry.metadata));
    }
    Ok(records)
}

async fn directory(folder: &Path, include_special: bool) -> BuiltinResult<Vec<Record>> {
    let absolute = absolute_folder(folder).await?;
    let metadata = vfs::metadata_async(folder).await.ok();
    let mut records = Vec::new();
    if include_special {
        records.push(Record::special(".", &absolute, metadata.as_ref()));
        records.push(Record::special("..", &absolute, metadata.as_ref()));
    }
    let entries = super::super::enumeration::directory(folder)
        .await
        .map_err(|error| {
            super::error::operation(format!("dir: unable to access '{absolute}' ({error})"))
        })?;
    records.extend(entries.into_iter().map(|entry| {
        let name = super::super::enumeration::filename(&entry.path)
            .to_string_lossy()
            .into_owned();
        Record::from_metadata(absolute.clone(), name, &entry.metadata)
    }));
    Ok(records)
}

fn current_directory() -> BuiltinResult<PathBuf> {
    vfs::current_dir().map_err(|error| {
        super::error::operation(format!(
            "dir: unable to determine current directory ({error})"
        ))
    })
}

async fn absolute_folder(path: &Path) -> BuiltinResult<String> {
    let joined = if super::super::super::is_rooted_path(path) {
        path.to_path_buf()
    } else {
        current_directory()?.join(path)
    };
    let normalized = vfs::canonicalize_async(&joined).await.unwrap_or(joined);
    Ok(path_to_string(&normalized))
}
