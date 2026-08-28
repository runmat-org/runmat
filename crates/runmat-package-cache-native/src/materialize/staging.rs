use super::download::read_verified;
use crate::filesystem::atomic_write;
use crate::NativeCacheError;
use runmat_package_cache::{CacheBackend, TreeEntryKind, TreeManifest};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_STAGING: AtomicU64 = AtomicU64::new(0);

pub(super) struct StagingTree {
    path: PathBuf,
    armed: bool,
}

impl StagingTree {
    pub(super) fn create(staging_root: &Path) -> Result<Self, NativeCacheError> {
        for _ in 0..100 {
            let sequence = NEXT_STAGING.fetch_add(1, Ordering::Relaxed);
            let path = staging_root.join(format!("tree-{}-{sequence}.partial", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, armed: true }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(NativeCacheError::io(path, error)),
            }
        }
        Err(NativeCacheError::Config(
            "could not allocate a unique staging directory".to_string(),
        ))
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingTree {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

pub(super) async fn assemble<B: CacheBackend>(
    backend: &B,
    staging: &Path,
    tree: &TreeManifest,
) -> Result<(), NativeCacheError> {
    for entry in tree
        .entries
        .iter()
        .filter(|entry| entry.kind == TreeEntryKind::Directory)
    {
        create_directory(&staging.join(entry.path.as_str()))?;
    }
    for entry in tree
        .entries
        .iter()
        .filter(|entry| entry.kind == TreeEntryKind::File)
    {
        let path = staging.join(entry.path.as_str());
        let parent = path
            .parent()
            .expect("normalized file path has staging parent");
        create_directory(parent)?;
        let digest = entry.digest.as_ref().expect("validated file has digest");
        let bytes = read_verified(backend, digest, entry.byte_len).await?;
        atomic_write(&path, &bytes)?;
        set_executable(&path, entry.executable)?;
    }
    for entry in tree
        .entries
        .iter()
        .filter(|entry| entry.kind == TreeEntryKind::Symlink)
    {
        let path = staging.join(entry.path.as_str());
        let parent = path
            .parent()
            .expect("normalized link path has staging parent");
        create_directory(parent)?;
        let target = entry
            .link_target
            .as_ref()
            .expect("validated symlink has target");
        create_symlink(
            &relative_target(
                parent.strip_prefix(staging).unwrap_or(Path::new("")),
                target.as_str(),
            ),
            &path,
            staging.join(target.as_str()).is_dir(),
        )?;
    }
    Ok(())
}

fn create_directory(path: &Path) -> Result<(), NativeCacheError> {
    std::fs::create_dir_all(path).map_err(|error| NativeCacheError::io(path, error))
}

fn relative_target(parent: &Path, target: &str) -> PathBuf {
    let parent_components: Vec<_> = parent.components().collect();
    let target_path = Path::new(target);
    let target_components: Vec<_> = target_path.components().collect();
    let common = parent_components
        .iter()
        .zip(&target_components)
        .take_while(|(left, right)| left == right)
        .count();
    let mut relative = PathBuf::new();
    for _ in common..parent_components.len() {
        relative.push("..");
    }
    for component in &target_components[common..] {
        relative.push(component.as_os_str());
    }
    relative
}

#[cfg(unix)]
fn create_symlink(target: &Path, path: &Path, _directory: bool) -> Result<(), NativeCacheError> {
    std::os::unix::fs::symlink(target, path).map_err(|error| NativeCacheError::io(path, error))
}

#[cfg(windows)]
fn create_symlink(target: &Path, path: &Path, directory: bool) -> Result<(), NativeCacheError> {
    let result = if directory {
        std::os::windows::fs::symlink_dir(target, path)
    } else {
        std::os::windows::fs::symlink_file(target, path)
    };
    result.map_err(|error| NativeCacheError::io(path, error))
}

#[cfg(unix)]
fn set_executable(path: &Path, executable: bool) -> Result<(), NativeCacheError> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = std::fs::metadata(path).map_err(|error| NativeCacheError::io(path, error))?;
    let mut permissions = metadata.permissions();
    let mode = if executable {
        permissions.mode() | 0o111
    } else {
        permissions.mode() & !0o111
    };
    permissions.set_mode(mode);
    std::fs::set_permissions(path, permissions).map_err(|error| NativeCacheError::io(path, error))
}

#[cfg(windows)]
fn set_executable(_path: &Path, _executable: bool) -> Result<(), NativeCacheError> {
    Ok(())
}
