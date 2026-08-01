use super::resolve::NativeResolvedProject;
use anyhow::{Context, Result};
use runmat_package::SourceId;
use runmat_package_cache::load_git_snapshot;
use runmat_package_cache_native::materialize::materialize_tree;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct VendorManifest {
    schema_version: u32,
    graph_digest: String,
    packages: Vec<VendoredPackage>,
}

#[derive(Serialize)]
struct VendoredPackage {
    identity: String,
    source: SourceId,
    path: String,
}

pub(super) async fn vendor(project: &NativeResolvedProject, output: &Path) -> Result<()> {
    let output = if output.is_absolute() {
        output.to_path_buf()
    } else {
        project.resolved.frozen.workspace_root.join(output)
    };
    std::fs::create_dir_all(&output)
        .with_context(|| format!("failed to create vendor directory {}", output.display()))?;
    let layout = project.cache_config.layout();
    let mut packages = Vec::new();
    for (identity, package) in &project.resolved.frozen.graph.packages {
        if identity == &project.resolved.frozen.graph.root {
            continue;
        }
        let source_root = match &package.instance.source {
            SourceId::Path(source) => project
                .resolved
                .frozen
                .workspace_root
                .join(source.workspace_path.as_str()),
            SourceId::Git(source) => {
                let snapshot = load_git_snapshot(&project.backend, source.clone())
                    .await
                    .with_context(|| format!("Git source {} is not cached", source.tree_digest))?;
                materialize_tree(&project.backend, &layout, &snapshot.tree)
                    .await
                    .context("failed to materialize vendored Git tree")?
            }
            _ => continue,
        };
        let directory = format!(
            "{}-{}",
            package.local_name,
            identity.to_string().replace(':', "_")
        );
        let destination = output.join(&directory);
        let canonical_root = std::fs::canonicalize(&source_root).with_context(|| {
            format!(
                "failed to canonicalize vendor source {}",
                source_root.display()
            )
        })?;
        if std::fs::symlink_metadata(&destination).is_ok() {
            if !trees_equal(&source_root, &destination)? {
                anyhow::bail!(
                    "vendor destination {} already exists with different content",
                    destination.display()
                );
            }
        } else {
            copy_tree(&source_root, &destination, &canonical_root)?;
        }
        packages.push(VendoredPackage {
            identity: identity.to_string(),
            source: package.instance.source.clone(),
            path: directory,
        });
    }
    packages.sort_by(|left, right| left.identity.cmp(&right.identity));
    let manifest = VendorManifest {
        schema_version: 1,
        graph_digest: project.resolved.frozen.graph.graph_digest.to_string(),
        packages,
    };
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    let manifest_path = output.join("runmat-vendor.json");
    if std::fs::symlink_metadata(&manifest_path)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        anyhow::bail!(
            "refusing to replace symlinked vendor manifest {}",
            manifest_path.display()
        );
    }
    std::fs::write(&manifest_path, bytes).context("failed to write vendor manifest")?;
    println!("Vendored package closure to {}", output.display());
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path, allowed_root: &Path) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(&source_path)?;
        if metadata.file_type().is_symlink() {
            let resolved = std::fs::canonicalize(&source_path).with_context(|| {
                format!(
                    "vendored symlink {} is broken or cannot be resolved",
                    source_path.display()
                )
            })?;
            if !resolved.starts_with(allowed_root) {
                anyhow::bail!(
                    "vendored symlink {} escapes package root {}",
                    source_path.display(),
                    allowed_root.display()
                );
            }
            copy_symlink(&source_path, &destination_path)?;
        } else if metadata.is_dir() {
            copy_tree(&source_path, &destination_path, allowed_root)?;
        } else if metadata.is_file() {
            std::fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn trees_equal(source: &Path, destination: &Path) -> Result<bool> {
    let mut source_entries = std::fs::read_dir(source)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut destination_entries = std::fs::read_dir(destination)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<std::io::Result<Vec<_>>>()?;
    source_entries.sort();
    destination_entries.sort();
    if source_entries != destination_entries {
        return Ok(false);
    }
    for name in source_entries {
        let source_path = source.join(&name);
        let destination_path = destination.join(name);
        let source_metadata = std::fs::symlink_metadata(&source_path)?;
        let destination_metadata = match std::fs::symlink_metadata(&destination_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        if source_metadata.file_type().is_symlink() || destination_metadata.file_type().is_symlink()
        {
            if !source_metadata.file_type().is_symlink()
                || !destination_metadata.file_type().is_symlink()
                || std::fs::read_link(&source_path)? != std::fs::read_link(&destination_path)?
            {
                return Ok(false);
            }
        } else if source_metadata.is_dir() && destination_metadata.is_dir() {
            if !trees_equal(&source_path, &destination_path)? {
                return Ok(false);
            }
        } else if source_metadata.is_file() && destination_metadata.is_file() {
            if std::fs::read(&source_path)? != std::fs::read(&destination_path)? {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(unix)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    let target = std::fs::read_link(source)?;
    std::os::unix::fs::symlink(target, destination)?;
    Ok(())
}

#[cfg(windows)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    let target = std::fs::read_link(source)?;
    if std::fs::metadata(source)?.is_dir() {
        std::os::windows::fs::symlink_dir(target, destination)?;
    } else {
        std::os::windows::fs::symlink_file(target, destination)?;
    }
    Ok(())
}
