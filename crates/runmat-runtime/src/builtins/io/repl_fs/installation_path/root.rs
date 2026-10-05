use std::path::{Path, PathBuf};

pub(crate) fn resolve() -> PathBuf {
    select(
        crate::builtins::common::env::var("RUNMAT_ROOT")
            .ok()
            .map(PathBuf::from),
        executable_path(),
        runmat_filesystem::current_dir().ok(),
    )
}

/// Factory search-path contribution known independently of the implicit
/// current directory. The executable/configured installation root is captured
/// by the session; an absent root is not replaced by ".".
pub(crate) fn configured_factory_root(cwd: &Path) -> Option<PathBuf> {
    let configured = crate::builtins::common::env::var("RUNMAT_ROOT")
        .ok()
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty());
    select_factory(configured, executable_path(), cwd)
}

fn select_factory(
    configured: Option<PathBuf>,
    executable: Option<PathBuf>,
    cwd: &Path,
) -> Option<PathBuf> {
    configured
        .map(|path| {
            if runmat_filesystem::current_provider()
                .path_namespace()
                .is_absolute(&path)
            {
                path
            } else {
                cwd.join(path)
            }
        })
        .or_else(|| executable.and_then(|path| path.parent().map(Path::to_path_buf)))
}

fn select(
    configured: Option<PathBuf>,
    executable: Option<PathBuf>,
    current: Option<PathBuf>,
) -> PathBuf {
    configured
        .filter(|path| !path.as_os_str().is_empty())
        .or_else(|| executable.and_then(|path| path.parent().map(Path::to_path_buf)))
        .or(current)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(not(target_arch = "wasm32"))]
fn executable_path() -> Option<PathBuf> {
    std::env::current_exe().ok()
}

#[cfg(target_arch = "wasm32")]
fn executable_path() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_root_has_precedence() {
        assert_eq!(
            select(
                Some(PathBuf::from("configured")),
                Some(PathBuf::from("bin/runmat")),
                Some(PathBuf::from("working")),
            ),
            PathBuf::from("configured")
        );
    }

    #[test]
    fn executable_parent_precedes_working_directory() {
        assert_eq!(
            select(
                None,
                Some(PathBuf::from("install/bin/runmat")),
                Some(PathBuf::from("working")),
            ),
            PathBuf::from("install/bin")
        );
    }

    #[test]
    fn working_directory_and_dot_are_ordered_fallbacks() {
        assert_eq!(
            select(None, None, Some(PathBuf::from("working"))),
            PathBuf::from("working")
        );
        assert_eq!(select(None, None, None), PathBuf::from("."));
    }

    #[test]
    fn configured_factory_root_never_substitutes_implicit_cwd() {
        #[cfg(windows)]
        let cwd = Path::new("C:\\working");
        #[cfg(not(windows))]
        let cwd = Path::new("/working");
        assert_eq!(select_factory(None, None, cwd), None);
        assert_eq!(
            select_factory(Some(PathBuf::from("configured")), None, cwd),
            Some(cwd.join("configured"))
        );
    }
}
