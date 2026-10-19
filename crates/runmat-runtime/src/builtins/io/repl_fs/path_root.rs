use std::path::Path;

pub(crate) fn is_rooted_path(path: &Path) -> bool {
    path.is_absolute() || path.has_root()
}

#[cfg(test)]
mod tests {
    use super::is_rooted_path;
    use std::path::Path;

    #[test]
    fn distinguishes_rooted_and_relative_paths() {
        assert!(is_rooted_path(Path::new("/")));
        assert!(!is_rooted_path(Path::new("relative/path")));
    }
}
