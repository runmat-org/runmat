/// File extensions recognized as MEX modules, in current MATLAB platform
/// families. The native suffix for the running target is selected separately
/// by [`mex_suffix`].
pub const MEX_EXTENSIONS: &[&str] = &[
    ".mexmaca64",
    ".mexmaci64",
    ".mexa64",
    ".mexw64",
    ".mexglx",
    ".mexw32",
    ".mexmaci",
    ".mex",
];

/// Return the MATLAB-compatible MEX suffix for a supported native target.
pub const fn mex_suffix() -> Option<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("mexmaca64")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("mexmaci64")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("mexa64")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("mexw64")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_test_target_has_an_explicit_suffix_policy() {
        assert!(mex_suffix().is_some());
        assert!(MEX_EXTENSIONS
            .iter()
            .any(|extension| extension.strip_prefix('.') == mex_suffix()));
    }
}
