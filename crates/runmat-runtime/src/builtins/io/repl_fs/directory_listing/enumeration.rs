use glob::Pattern;
use runmat_filesystem::{self as vfs, FsMetadata};
use std::collections::HashSet;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

pub(super) struct ListedEntry {
    pub path: PathBuf,
    pub metadata: FsMetadata,
    pub recursable: bool,
}

pub(super) async fn directory(path: &Path) -> io::Result<Vec<ListedEntry>> {
    let mut listed = Vec::new();
    for entry in vfs::read_dir_async(path).await? {
        let entry_path = entry.path().to_path_buf();
        let recursable = entry.is_dir();
        let metadata = match vfs::metadata_async(&entry_path).await {
            Ok(metadata) => metadata,
            Err(_) => vfs::symlink_metadata_async(&entry_path).await?,
        };
        listed.push(ListedEntry {
            path: entry_path,
            metadata,
            recursable,
        });
    }
    Ok(listed)
}

pub(super) async fn matching(pattern: &Path) -> io::Result<Vec<ListedEntry>> {
    let (base, components) = split_pattern(pattern)?;
    let mut pending = vec![(base, 0usize)];
    let mut paths = HashSet::new();

    while let Some((folder, index)) = pending.pop() {
        if index == components.len() {
            paths.insert(folder);
            continue;
        }
        let component = &components[index];
        if component == "**" {
            let entries = directory(&folder).await?;
            if index + 1 == components.len() {
                for entry in entries {
                    let path = child_path(&folder, &entry.path);
                    if entry.recursable {
                        pending.push((path.clone(), index));
                    }
                    paths.insert(path);
                }
            } else {
                for entry in entries {
                    if entry.recursable {
                        pending.push((child_path(&folder, &entry.path), index));
                    }
                }
                pending.push((folder, index + 1));
            }
            continue;
        }

        let matcher = Pattern::new(component).map_err(invalid_pattern)?;
        for entry in directory(&folder).await? {
            let name = entry.path.file_name().and_then(|name| name.to_str());
            if !name.is_some_and(|name| matcher.matches(name)) {
                continue;
            }
            let path = child_path(&folder, &entry.path);
            if index + 1 == components.len() {
                paths.insert(path);
            } else if entry.recursable {
                pending.push((path, index + 1));
            }
        }
    }

    let mut listed = Vec::with_capacity(paths.len());
    for path in paths {
        let metadata = match vfs::metadata_async(&path).await {
            Ok(metadata) => metadata,
            Err(_) => vfs::symlink_metadata_async(&path).await?,
        };
        listed.push(ListedEntry {
            path,
            metadata,
            recursable: false,
        });
    }
    Ok(listed)
}

fn child_path(folder: &Path, provider_path: &Path) -> PathBuf {
    folder.join(filename(provider_path))
}

fn split_pattern(pattern: &Path) -> io::Result<(PathBuf, Vec<String>)> {
    let mut base = PathBuf::new();
    let mut patterns = Vec::new();
    let mut found_pattern = false;

    for component in pattern.components() {
        match component {
            Component::Prefix(_) | Component::RootDir if !found_pattern => base.push(component),
            Component::CurDir if !found_pattern => {}
            Component::ParentDir if !found_pattern => base.push(".."),
            Component::Normal(value) => {
                let text = value.to_string_lossy().into_owned();
                if found_pattern || contains_wildcards(&text) {
                    found_pattern = true;
                    patterns.push(text);
                } else {
                    base.push(value);
                }
            }
            _ => return Err(invalid_pattern("path root appears after wildcard")),
        }
    }
    if !found_pattern {
        return Err(invalid_pattern("pattern contains no wildcard"));
    }
    if base.as_os_str().is_empty() {
        base = PathBuf::from(".");
    }
    Ok((base, patterns))
}

fn contains_wildcards(text: &str) -> bool {
    text.contains(['*', '?'])
}

fn invalid_pattern(error: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error.to_string())
}

pub(super) fn filename(path: &Path) -> OsString {
    path.file_name()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.as_os_str().to_owned())
}

#[cfg(test)]
mod tests;
