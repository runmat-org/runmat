use super::resolve::NativeResolvedProject;
use anyhow::{Context, Result};
use runmat_package::{
    NormalizedRelativePath, SourceId, VendorManifest, VendoredPackage, VENDOR_MANIFEST_FILENAME,
};
use runmat_package_cache::load_git_snapshot;
use runmat_package_cache_native::materialize::materialize_tree;
use std::path::Path;

pub(super) async fn vendor(project: &NativeResolvedProject, output: &Path) -> Result<()> {
    let requested_output = if output.is_absolute() {
        output.to_path_buf()
    } else {
        project.resolved.frozen.workspace_root.join(output)
    };
    let lexical_relative = requested_output
        .strip_prefix(&project.resolved.frozen.workspace_root)
        .with_context(|| {
            format!(
                "vendor directory {} must remain inside workspace {}",
                requested_output.display(),
                project.resolved.frozen.workspace_root.display()
            )
        })?;
    let lexical_relative = NormalizedRelativePath::new(lexical_relative)
        .context("vendor output must be a portable project-relative path")?;
    let output = project
        .resolved
        .frozen
        .workspace_root
        .join(lexical_relative.as_str());
    ensure_existing_ancestor_is_inside(&output, &project.resolved.frozen.workspace_root)?;
    std::fs::create_dir_all(&output)
        .with_context(|| format!("failed to create vendor directory {}", output.display()))?;
    let output = std::fs::canonicalize(&output).with_context(|| {
        format!(
            "failed to canonicalize vendor directory {}",
            output.display()
        )
    })?;
    let relative_output = output
        .strip_prefix(&project.resolved.frozen.workspace_root)
        .with_context(|| {
            format!(
                "vendor directory {} must remain inside workspace {}",
                output.display(),
                project.resolved.frozen.workspace_root.display()
            )
        })?;
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
        if std::fs::symlink_metadata(&destination)
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            anyhow::bail!(
                "refusing to use symlinked vendor destination {}",
                destination.display()
            );
        } else if std::fs::symlink_metadata(&destination).is_ok() {
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
            identity: identity.clone(),
            source: package.instance.source.clone(),
            path: NormalizedRelativePath::new(relative_output.join(directory))
                .context("vendor package path is not portable")?,
        });
    }
    let manifest = VendorManifest::new(project.resolved.lock.graph_digest.clone(), packages)
        .map_err(anyhow::Error::msg)?;
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    let manifest_path = project
        .resolved
        .frozen
        .workspace_root
        .join(VENDOR_MANIFEST_FILENAME);
    if std::fs::symlink_metadata(&manifest_path)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        anyhow::bail!(
            "refusing to replace symlinked vendor manifest {}",
            manifest_path.display()
        );
    }
    let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).context("failed to stage vendor manifest")?;
    use std::io::Write as _;
    temporary
        .write_all(&bytes)
        .context("failed to write staged vendor manifest")?;
    temporary
        .as_file()
        .sync_all()
        .context("failed to sync staged vendor manifest")?;
    temporary
        .persist(&manifest_path)
        .map_err(|error| error.error)
        .context("failed to atomically publish vendor manifest")?;
    println!(
        "Vendored package closure to {} and wrote {}",
        output.display(),
        manifest_path.display()
    );
    Ok(())
}

fn ensure_existing_ancestor_is_inside(path: &Path, workspace: &Path) -> Result<()> {
    let mut ancestor = path;
    while std::fs::symlink_metadata(ancestor).is_err() {
        ancestor = ancestor.parent().ok_or_else(|| {
            anyhow::anyhow!("vendor output {} has no existing ancestor", path.display())
        })?;
    }
    let canonical = std::fs::canonicalize(ancestor).with_context(|| {
        format!(
            "failed to canonicalize vendor output ancestor {}",
            ancestor.display()
        )
    })?;
    if !canonical.starts_with(workspace) {
        anyhow::bail!(
            "vendor output ancestor {} resolves outside workspace {}",
            ancestor.display(),
            workspace.display()
        );
    }
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
