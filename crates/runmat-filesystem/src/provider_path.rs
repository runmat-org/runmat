use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::{provider_state_lock, FsProvider};

/// Absolute-path grammar owned by a filesystem provider. A rooted virtual
/// path is independent of process CWD even on native Windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsPathNamespace {
    NativeOs,
    VirtualRooted,
}

impl FsPathNamespace {
    pub fn is_absolute(self, path: &Path) -> bool {
        match self {
            Self::NativeOs => path.is_absolute(),
            Self::VirtualRooted => {
                path.has_root() && !matches!(path.components().next(), Some(Component::Prefix(_)))
            }
        }
    }
}

fn snapshot_provider() -> io::Result<(Arc<dyn FsProvider>, Option<PathBuf>)> {
    if let Some((provider, directory)) = crate::invocation::snapshot() {
        return Ok((crate::captured_read::selected_view(&provider)?, directory));
    }
    if let Some(selected) = crate::captured_read::snapshot() {
        return Ok(selected);
    }
    let state = provider_state_lock()
        .read()
        .expect("filesystem provider lock poisoned");
    Ok((state.provider.clone(), state.current_dir_override.clone()))
}

fn captured_cwd(virtual_dir: Option<PathBuf>) -> io::Result<PathBuf> {
    if let Some(dir) = virtual_dir {
        return Ok(dir);
    }
    if crate::captured_read::snapshot().is_some() {
        return Err(crate::captured_read::missing_directory());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::current_dir()
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(PathBuf::from("/"))
    }
}

fn resolve_with_cwd(
    namespace: FsPathNamespace,
    path: &Path,
    virtual_dir: Option<PathBuf>,
    fallback: impl FnOnce() -> io::Result<PathBuf>,
) -> io::Result<PathBuf> {
    if namespace.is_absolute(path) {
        return Ok(path.to_path_buf());
    }
    Ok(virtual_dir.map_or_else(fallback, Ok)?.join(path))
}

/// Capture path and provider as one authority. Provider callbacks and process
/// CWD lookup occur after releasing the registry lock. Absolute paths need no
/// working-directory service.
pub fn capture_provider_path(path: &Path) -> io::Result<(Arc<dyn FsProvider>, PathBuf)> {
    let (provider, virtual_dir) = snapshot_provider()?;
    let resolved = resolve_with_cwd(provider.path_namespace(), path, virtual_dir, || {
        captured_cwd(None)
    })?;
    Ok((provider, resolved))
}

/// Capture one provider and only the CWD demanded by the known path set.
/// Callers with a path materialized later must explicitly capture CWD then.
pub fn capture_provider_for_paths(
    paths: &[&Path],
) -> io::Result<(Arc<dyn FsProvider>, Option<PathBuf>)> {
    let (provider, virtual_dir) = snapshot_provider()?;
    let namespace = provider.path_namespace();
    let cwd = if paths.iter().all(|path| namespace.is_absolute(path)) {
        None
    } else {
        Some(captured_cwd(virtual_dir)?)
    };
    Ok((provider, cwd))
}

pub(super) fn capture_provider_pair(
    from: &Path,
    to: &Path,
) -> io::Result<(Arc<dyn FsProvider>, PathBuf, PathBuf)> {
    let (provider, virtual_dir) = snapshot_provider()?;
    let namespace = provider.path_namespace();
    let cwd = if namespace.is_absolute(from) && namespace.is_absolute(to) {
        None
    } else {
        Some(captured_cwd(virtual_dir)?)
    };
    let resolve = |path: &Path| {
        if namespace.is_absolute(path) {
            path.to_path_buf()
        } else {
            cwd.as_ref()
                .expect("relative path requires captured CWD")
                .join(path)
        }
    };
    Ok((provider, resolve(from), resolve(to)))
}

pub(super) fn capture_provider_many(
    paths: &[PathBuf],
) -> io::Result<(Arc<dyn FsProvider>, Vec<PathBuf>)> {
    let (provider, virtual_dir) = snapshot_provider()?;
    let namespace = provider.path_namespace();
    let cwd = if paths.iter().all(|path| namespace.is_absolute(path)) {
        None
    } else {
        Some(captured_cwd(virtual_dir)?)
    };
    let resolved = paths
        .iter()
        .map(|path| {
            if namespace.is_absolute(path) {
                path.clone()
            } else {
                cwd.as_ref()
                    .expect("relative path requires captured CWD")
                    .join(path)
            }
        })
        .collect();
    Ok((provider, resolved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{provider_override_lock, replace_provider, File, MemoryFsProvider};
    use std::io::{ErrorKind, Read};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn relative_paths_require_working_directory_capture() {
        for path in ["", ".", "data.bin", "folder/data.bin"] {
            assert!(!FsPathNamespace::NativeOs.is_absolute(Path::new(path)));
            assert!(!FsPathNamespace::VirtualRooted.is_absolute(Path::new(path)));
        }
    }

    #[cfg(all(not(windows), not(target_arch = "wasm32")))]
    #[test]
    fn rooted_provider_paths_are_absolute() {
        for path in ["/", "/data.bin", "/folder/data.bin"] {
            assert!(FsPathNamespace::NativeOs.is_absolute(Path::new(path)));
            assert!(FsPathNamespace::VirtualRooted.is_absolute(Path::new(path)));
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn browser_virtual_roots_do_not_depend_on_os_absolute_rules() {
        assert!(FsPathNamespace::VirtualRooted.is_absolute(Path::new("/data.bin")));
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_require_drive_or_network_qualification() {
        assert_eq!(
            MemoryFsProvider::new().path_namespace(),
            FsPathNamespace::VirtualRooted
        );
        assert_eq!(
            crate::NativeFsProvider.path_namespace(),
            FsPathNamespace::NativeOs
        );
        assert!(!FsPathNamespace::NativeOs.is_absolute(Path::new(r"\folder\data.bin")));
        assert!(FsPathNamespace::VirtualRooted.is_absolute(Path::new(r"\folder\data.bin")));
        assert!(FsPathNamespace::NativeOs.is_absolute(Path::new(r"C:\folder\data.bin")));
        assert!(FsPathNamespace::NativeOs.is_absolute(Path::new(r"\\server\share\data.bin")));
        assert!(!FsPathNamespace::VirtualRooted.is_absolute(Path::new(r"C:\folder\data.bin")));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn captured_virtual_root_keeps_original_provider_after_swap() {
        let _lock = provider_override_lock();
        let first = Arc::new(MemoryFsProvider::with_current_dir("/reports"));
        let second = Arc::new(MemoryFsProvider::with_current_dir("/reports"));
        first
            .write_project_path("/reports/data.txt", b"first")
            .unwrap();
        second
            .write_project_path("/reports/data.txt", b"second")
            .unwrap();
        let _first = replace_provider(first.clone());
        let (provider, path) = capture_provider_path(Path::new("/reports/data.txt")).unwrap();
        let first_dyn: Arc<dyn FsProvider> = first.clone();
        assert!(Arc::ptr_eq(&provider, &first_dyn));
        let _second = replace_provider(second);
        let mut file = File::open_on(provider.as_ref(), &path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"first");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn captured_open_rejects_relative_path_before_provider_effect() {
        let provider = MemoryFsProvider::with_current_dir("/reports");
        assert_eq!(
            File::open_on(&provider, Path::new("data.txt"))
                .expect_err("relative captured path is invalid")
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            futures::executor::block_on(File::open_async_on(&provider, Path::new("data.txt")))
                .expect_err("relative captured path is invalid")
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rooted_virtual_path_does_not_request_unavailable_cwd() {
        let root = Path::new("/data.txt");
        let resolved = resolve_with_cwd(FsPathNamespace::VirtualRooted, root, None, || {
            Err(io::Error::other("CWD is unavailable"))
        })
        .expect("rooted path does not need CWD");
        assert_eq!(resolved, root);
        assert_eq!(
            resolve_with_cwd(
                FsPathNamespace::VirtualRooted,
                Path::new("data.txt"),
                None,
                || { Err(io::Error::other("CWD is unavailable")) }
            )
            .unwrap_err()
            .to_string(),
            "CWD is unavailable"
        );
    }
}
