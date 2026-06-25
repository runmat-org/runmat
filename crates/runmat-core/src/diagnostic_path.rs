use std::path::{Path, PathBuf};

pub(crate) fn display_path_for_current_cwd(path: &Path) -> String {
    let Ok(cwd) = runmat_filesystem::current_dir() else {
        return path_to_string(path);
    };
    display_path_from_base(path, &cwd)
}

pub(crate) fn display_path_from_base(path: &Path, base: &Path) -> String {
    relative_display_path(path, base)
        .map(|path| path_to_string(&path))
        .unwrap_or_else(|| path_to_string(path))
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

fn relative_display_path(path: &Path, base: &Path) -> Option<PathBuf> {
    let relative = path
        .strip_prefix(base)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .or_else(|| canonical_relative_display_path(path, base));

    #[cfg(windows)]
    {
        relative.or_else(|| windows_relative_display_path(path, base))
    }

    #[cfg(not(windows))]
    {
        relative
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn canonical_relative_display_path(path: &Path, base: &Path) -> Option<PathBuf> {
    if !path.is_absolute() || !base.is_absolute() {
        return None;
    }

    let path = path.canonicalize().ok()?;
    let base = base.canonicalize().ok()?;
    path.strip_prefix(base)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(Path::to_path_buf)
}

#[cfg(target_arch = "wasm32")]
fn canonical_relative_display_path(_path: &Path, _base: &Path) -> Option<PathBuf> {
    None
}

#[cfg(windows)]
fn windows_relative_display_path(path: &Path, base: &Path) -> Option<PathBuf> {
    let path = windows_display_key(path);
    let base = windows_display_key(base);
    if path.eq_ignore_ascii_case(&base) {
        return None;
    }

    let prefix = if base.ends_with('\\') {
        base
    } else {
        format!("{base}\\")
    };
    let path_lower = path.to_ascii_lowercase();
    let prefix_lower = prefix.to_ascii_lowercase();
    path_lower
        .starts_with(&prefix_lower)
        .then(|| PathBuf::from(&path[prefix.len()..]))
        .filter(|relative| !relative.as_os_str().is_empty())
}

#[cfg(windows)]
fn windows_display_key(path: &Path) -> String {
    let mut text = path.to_string_lossy().replace('/', "\\");
    if let Some(stripped) = text.strip_prefix(r"\\?\UNC\") {
        text = format!(r"\\{stripped}");
    } else if let Some(stripped) = text.strip_prefix(r"\\?\") {
        text = stripped.to_string();
    }
    while text.ends_with('\\') && text.len() > 1 && !is_windows_drive_root(&text) {
        text.pop();
    }
    text
}

#[cfg(windows)]
fn is_windows_drive_root(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() == 3 && bytes[1] == b':' && bytes[2] == b'\\'
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

    #[cfg(windows)]
    #[test]
    fn display_path_strips_extended_length_base_equivalent() {
        let base = Path::new(r"C:\project");
        let path = Path::new(r"\\?\C:\project\src\main.m");
        let expected = Path::new("src").join("main.m");

        assert_eq!(PathBuf::from(display_path_from_base(path, base)), expected);
    }

    #[cfg(windows)]
    #[test]
    fn display_path_strips_virtual_root_base() {
        let base = Path::new("/");
        let path = Path::new("/main.m");

        assert_eq!(
            PathBuf::from(display_path_from_base(path, base)),
            PathBuf::from("main.m")
        );
    }
}
