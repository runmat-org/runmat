use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EnvironmentAllowlist {
    names: BTreeSet<String>,
}

impl EnvironmentAllowlist {
    pub fn new(names: impl IntoIterator<Item = String>) -> Self {
        Self {
            names: names.into_iter().collect(),
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        #[cfg(windows)]
        {
            self.names
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(name))
        }
        #[cfg(not(windows))]
        {
            self.names.contains(name)
        }
    }

    /// The ambient variables required to start a child with the same native
    /// runtime linkage as its parent.
    ///
    /// This is intentionally narrower than inheriting the application
    /// environment. It preserves only platform loader and process-bootstrap
    /// state; callers must still add every variable exposed to the child
    /// application explicitly.
    pub fn platform_runtime() -> Self {
        #[cfg(target_os = "linux")]
        const NAMES: &[&str] = &["LD_LIBRARY_PATH", "LD_PRELOAD"];
        #[cfg(target_os = "macos")]
        const NAMES: &[&str] = &[
            "DYLD_LIBRARY_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
            "DYLD_FALLBACK_FRAMEWORK_PATH",
            "DYLD_INSERT_LIBRARIES",
        ];
        #[cfg(windows)]
        const NAMES: &[&str] = &["PATH", "PATHEXT", "SystemRoot", "WINDIR"];
        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        const NAMES: &[&str] = &[];

        Self::new(NAMES.iter().map(|name| (*name).to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::EnvironmentAllowlist;

    #[test]
    fn platform_runtime_is_narrower_than_application_environment() {
        let allowlist = EnvironmentAllowlist::platform_runtime();
        assert!(!allowlist.contains("HOME"));
        assert!(!allowlist.contains("RUNMAT_CONFIG"));

        #[cfg(target_os = "linux")]
        assert!(allowlist.contains("LD_LIBRARY_PATH"));
        #[cfg(target_os = "macos")]
        assert!(allowlist.contains("DYLD_LIBRARY_PATH"));
        #[cfg(windows)]
        assert!(allowlist.contains("SystemRoot"));
    }
}
