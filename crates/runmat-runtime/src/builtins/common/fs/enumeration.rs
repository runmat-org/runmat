use glob::Pattern;
use runmat_filesystem::{self as vfs, FsMetadata, FsProvider};
use std::collections::HashSet;
use std::ffi::OsString;
use std::future::Future;
use std::io;
use std::path::{Component, Path, PathBuf};

pub(crate) struct ListedEntry {
    pub path: PathBuf,
    pub metadata: FsMetadata,
    pub recursable: bool,
}

/// Liveness authority retained by an owner that enumerates through a captured
/// provider. The check is synchronous because it guards, rather than performs,
/// each provider operation.
pub(crate) trait EnumerationAuthority {
    fn check(&self) -> io::Result<()>;
}

struct UnrestrictedEnumeration;

impl EnumerationAuthority for UnrestrictedEnumeration {
    fn check(&self) -> io::Result<()> {
        Ok(())
    }
}

async fn guarded<T>(
    authority: &dyn EnumerationAuthority,
    work: impl Future<Output = io::Result<T>>,
) -> io::Result<T> {
    authority.check()?;
    let result = work.await;
    if result.is_ok() {
        authority.check()?;
    }
    result
}

pub(crate) async fn directory(path: &Path) -> io::Result<Vec<ListedEntry>> {
    let (provider, cwd) = vfs::current_provider_with_current_dir()?;
    directory_in(provider.as_ref(), &cwd, path).await
}

pub(crate) async fn directory_in(
    provider: &dyn FsProvider,
    cwd: &Path,
    path: &Path,
) -> io::Result<Vec<ListedEntry>> {
    directory_in_guarded(provider, cwd, path, &UnrestrictedEnumeration).await
}

pub(crate) async fn directory_in_guarded(
    provider: &dyn FsProvider,
    cwd: &Path,
    path: &Path,
    authority: &dyn EnumerationAuthority,
) -> io::Result<Vec<ListedEntry>> {
    let mut listed = Vec::new();
    for entry in guarded(authority, provider.read_dir(&resolve(provider, cwd, path))).await? {
        let entry_path = entry.path().to_path_buf();
        let recursable = entry.is_dir();
        let metadata = match guarded(
            authority,
            provider.metadata(&resolve(provider, cwd, &entry_path)),
        )
        .await
        {
            Ok(metadata) => metadata,
            Err(error) => {
                if authority.check().is_err() {
                    return Err(error);
                }
                guarded(
                    authority,
                    provider.symlink_metadata(&resolve(provider, cwd, &entry_path)),
                )
                .await?
            }
        };
        listed.push(ListedEntry {
            path: entry_path,
            metadata,
            recursable,
        });
    }
    Ok(listed)
}

// Wildcards do not match missing or non-directory literal prefixes. Inspect
// metadata first because providers need not classify read_dir(file) identically.
async fn matching_directory(
    provider: &dyn FsProvider,
    cwd: &Path,
    path: &Path,
    authority: &dyn EnumerationAuthority,
) -> io::Result<Vec<ListedEntry>> {
    match guarded(authority, provider.metadata(&resolve(provider, cwd, path))).await {
        Ok(metadata) if metadata.is_dir() => {
            directory_in_guarded(provider, cwd, path, authority).await
        }
        Ok(_) => Ok(Vec::new()),
        Err(error) if authority.check().is_err() => Err(error),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            Ok(Vec::new())
        }
        Err(error) => Err(error),
    }
}

pub(crate) async fn matching(pattern: &Path) -> io::Result<Vec<ListedEntry>> {
    let (provider, cwd) = vfs::current_provider_with_current_dir()?;
    matching_in(provider.as_ref(), &cwd, pattern).await
}

pub(crate) async fn matching_in(
    provider: &dyn FsProvider,
    cwd: &Path,
    pattern: &Path,
) -> io::Result<Vec<ListedEntry>> {
    matching_in_guarded(provider, cwd, pattern, &UnrestrictedEnumeration).await
}

pub(crate) async fn matching_in_guarded(
    provider: &dyn FsProvider,
    cwd: &Path,
    pattern: &Path,
    authority: &dyn EnumerationAuthority,
) -> io::Result<Vec<ListedEntry>> {
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
            let entries = matching_directory(provider, cwd, &folder, authority).await?;
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
        for entry in matching_directory(provider, cwd, &folder, authority).await? {
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
        let metadata =
            match guarded(authority, provider.metadata(&resolve(provider, cwd, &path))).await {
                Ok(metadata) => metadata,
                Err(error) => {
                    if authority.check().is_err() {
                        return Err(error);
                    }
                    guarded(
                        authority,
                        provider.symlink_metadata(&resolve(provider, cwd, &path)),
                    )
                    .await?
                }
            };
        listed.push(ListedEntry {
            path,
            metadata,
            recursable: false,
        });
    }
    Ok(listed)
}

fn resolve(provider: &dyn FsProvider, cwd: &Path, path: &Path) -> PathBuf {
    if provider.path_namespace().is_absolute(path) {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
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

pub(crate) fn filename(path: &Path) -> OsString {
    path.file_name()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.as_os_str().to_owned())
}

#[cfg(test)]
mod tests;
