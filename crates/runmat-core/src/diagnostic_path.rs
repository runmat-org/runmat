use std::path::{Path, PathBuf};

pub(crate) fn display_path_for_current_cwd(path: &Path) -> String {
    let Ok(cwd) = runmat_filesystem::current_dir() else {
        return path_to_string(path);
    };
    display_path_from_base(path, &cwd)
}

pub(crate) fn display_path_from_base(path: &Path, base: &Path) -> String {
    let display_path = if path.is_absolute() {
        path.strip_prefix(base)
            .ok()
            .filter(|relative| !relative.as_os_str().is_empty())
            .unwrap_or(path)
    } else {
        path
    };
    path_to_string(display_path)
}

pub(crate) fn resolve_against_base(path: &str, base: &Path) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn display_path_strips_base_for_nested_path() {
        let base = Path::new("/project");
        let path = Path::new("/project/src/main.m");
        let expected = Path::new("src").join("main.m");

        assert_eq!(
            display_path_from_base(path, base),
            path_to_string(&expected)
        );
    }

    #[cfg(unix)]
    #[test]
    fn display_path_leaves_external_absolute_path() {
        let base = Path::new("/project");
        let path = Path::new("/other/main.m");

        assert_eq!(display_path_from_base(path, base), path_to_string(path));
    }

    #[test]
    fn display_path_preserves_relative_path() {
        let base = Path::new("/project");
        let path = Path::new("src/main.m");

        assert_eq!(display_path_from_base(path, base), "src/main.m");
    }
}
