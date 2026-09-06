use runmat_filesystem as vfs;
use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::builtins::common::fs::{
    contains_wildcards, expand_user_path, path_to_string, sort_entries,
};
use crate::BuiltinResult;

use super::input::Input;

pub(super) async fn execute(input: Input) -> BuiltinResult<Vec<String>> {
    match input {
        Input::Current => directory(&current_directory()?).await,
        Input::Name(name) => from_text(&name).await,
    }
}

async fn from_text(text: &str) -> BuiltinResult<Vec<String>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return directory(&current_directory()?).await;
    }
    let expanded = expand_user_path(trimmed, "ls").map_err(super::error::operation)?;
    if contains_wildcards(&expanded) {
        pattern(Path::new(&expanded), trimmed).await
    } else {
        path(Path::new(&expanded), trimmed).await
    }
}

async fn path(path: &Path, display: &str) -> BuiltinResult<Vec<String>> {
    match vfs::metadata_async(path).await {
        Ok(metadata) if metadata.is_dir() => directory(path).await,
        Ok(_) => Ok(vec![path_to_string(path)]),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(super::error::operation(format!(
            "ls: unable to access '{display}' ({error})"
        ))),
    }
}

async fn directory(folder: &Path) -> BuiltinResult<Vec<String>> {
    let display = path_to_string(folder);
    let entries = super::super::enumeration::directory(folder)
        .await
        .map_err(|error| {
            super::error::operation(format!("ls: unable to access '{display}' ({error})"))
        })?;
    let mut rows = entries
        .into_iter()
        .map(|entry| {
            let mut name = super::super::enumeration::filename(&entry.path)
                .to_string_lossy()
                .into_owned();
            append_directory_suffix(&mut name, entry.metadata.is_dir());
            name
        })
        .collect::<Vec<_>>();
    sort_entries(&mut rows);
    Ok(rows)
}

async fn pattern(pattern: &Path, display: &str) -> BuiltinResult<Vec<String>> {
    let entries = super::super::enumeration::matching(pattern)
        .await
        .map_err(|error| {
            super::error::operation(format!(
                "ls: unable to enumerate matches for '{display}' ({error})"
            ))
        })?;
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for entry in entries {
        let display_path = entry
            .path
            .strip_prefix(Path::new("."))
            .unwrap_or(&entry.path);
        let mut name = path_to_string(display_path);
        append_directory_suffix(&mut name, entry.metadata.is_dir());
        if seen.insert(name.clone()) {
            rows.push(name);
        }
    }
    sort_entries(&mut rows);
    Ok(rows)
}

fn append_directory_suffix(text: &mut String, is_directory: bool) {
    if is_directory && !text.ends_with(std::path::MAIN_SEPARATOR) {
        text.push(std::path::MAIN_SEPARATOR);
    }
}

fn current_directory() -> BuiltinResult<PathBuf> {
    vfs::current_dir().map_err(|error| {
        super::error::operation(format!(
            "ls: unable to determine current directory ({error})"
        ))
    })
}
