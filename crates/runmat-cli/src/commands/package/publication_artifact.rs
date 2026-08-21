use anyhow::{bail, Context, Result};
use runmat_package_cache::{ArtifactEntryRole, PublicationEntry, PublicationPolicy};
use std::path::{Component, Path, PathBuf};

pub(super) fn collect_entries(
    root: &Path,
    policy: &PublicationPolicy,
) -> Result<Vec<PublicationEntry>> {
    let mut entries = Vec::new();
    collect_directory(root, root, policy, &mut entries)?;
    Ok(entries)
}

fn collect_directory(
    root: &Path,
    directory: &Path,
    policy: &PublicationPolicy,
    entries: &mut Vec<PublicationEntry>,
) -> Result<()> {
    let mut children = std::fs::read_dir(directory)
        .with_context(|| {
            format!(
                "failed to read publication directory {}",
                directory.display()
            )
        })?
        .collect::<Result<Vec<_>, _>>()?;
    children.sort_by_key(std::fs::DirEntry::file_name);
    for child in children {
        let path = child.path();
        let relative = path
            .strip_prefix(root)
            .expect("publication child remains beneath root");
        let relative = normalized_path(relative)?;
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("failed to inspect publication entry {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            if !policy.accepts_path(&relative, role(&path))? {
                continue;
            }
            let target = std::fs::read_link(&path).with_context(|| {
                format!("failed to read publication symlink {}", path.display())
            })?;
            entries.push(PublicationEntry::symlink(
                &relative,
                normalized_link_target(relative.as_str(), &target)?,
                role(&path),
            )?);
        } else if metadata.is_dir() {
            if !policy.should_descend(&relative) {
                continue;
            }
            if policy.accepts_path(&relative, ArtifactEntryRole::Resource)? {
                entries.push(PublicationEntry::directory(&relative)?);
            }
            collect_directory(root, &path, policy, entries)?;
        } else if metadata.is_file() {
            if !policy.accepts_path(&relative, role(&path))? {
                continue;
            }
            let bytes = std::fs::read(&path)
                .with_context(|| format!("failed to read publication file {}", path.display()))?;
            entries.push(PublicationEntry::file(
                &relative,
                bytes,
                executable(&metadata),
                role(&path),
            )?);
        }
    }
    Ok(())
}

fn normalized_path(path: &Path) -> Result<String> {
    let value = path
        .to_str()
        .context("publication paths must be valid UTF-8")?
        .replace('\\', "/");
    if value.is_empty() {
        bail!("publication path is empty");
    }
    Ok(value)
}

pub(super) fn normalized_link_target(link_path: &str, target: &Path) -> Result<String> {
    if target.is_absolute() {
        bail!("publication symlink `{link_path}` has an absolute target");
    }
    let mut resolved = PathBuf::from(link_path);
    resolved.pop();
    for component in target.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(value) => resolved.push(value),
            Component::ParentDir => {
                if !resolved.pop() {
                    bail!("publication symlink `{link_path}` escapes the package root");
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("publication symlink `{link_path}` has an absolute target");
            }
        }
    }
    normalized_path(&resolved)
}

fn role(path: &Path) -> ArtifactEntryRole {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "m" => ArtifactEntryRole::Source,
        "dll" | "dylib" | "so" | "mex" | "mexa64" | "mexmaci64" | "mexw64" => {
            ArtifactEntryRole::Native
        }
        _ => ArtifactEntryRole::Resource,
    }
}

#[cfg(unix)]
fn executable(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(_metadata: &std::fs::Metadata) -> bool {
    false
}
