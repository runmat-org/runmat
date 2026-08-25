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
    }
}
