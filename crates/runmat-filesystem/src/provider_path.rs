use std::path::Path;

/// Whether a provider path is independent of the current working directory.
/// Browser providers use a rooted virtual namespace; native providers retain
/// the host's absolute-path rules, including Windows drive qualification.
pub fn is_absolute_provider_path(path: &Path) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        path.has_root()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        path.is_absolute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn relative_paths_require_working_directory_capture() {
        for path in ["", ".", "data.bin", "folder/data.bin"] {
            assert!(!is_absolute_provider_path(Path::new(path)));
        }
    }

    #[cfg(not(windows))]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rooted_provider_paths_are_absolute() {
        for path in ["/", "/data.bin", "/folder/data.bin"] {
            assert!(is_absolute_provider_path(Path::new(path)));
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_require_drive_or_network_qualification() {
        assert!(!is_absolute_provider_path(Path::new(r"\folder\data.bin")));
        assert!(is_absolute_provider_path(Path::new(r"C:\folder\data.bin")));
        assert!(is_absolute_provider_path(Path::new(
            r"\\server\share\data.bin"
        )));
    }
}
