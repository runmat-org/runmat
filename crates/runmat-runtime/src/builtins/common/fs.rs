//! Filesystem helper utilities shared across REPL-facing builtins.
//!
//! These helpers centralize path normalization, wildcard detection,
//! and platform-aware sorting so builtins such as `ls` and `dir`
//! can focus on their user-facing semantics.

use std::cmp::Ordering;
use std::env;
use std::path::{Path, PathBuf};

/// Expand a user-relative path (e.g., `~` or `~/Documents`) into an absolute
/// filesystem path string. Returns the original string when no expansion is
/// required.
pub fn expand_user_path(raw: &str, error_prefix: &str) -> Result<String, String> {
    if raw == "~" {
        return home_directory()
            .map(|path| path_to_string(&path))
            .ok_or_else(|| format!("{error_prefix}: unable to resolve home directory"));
    }

    if let Some(stripped) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
        let home = home_directory()
            .ok_or_else(|| format!("{error_prefix}: unable to resolve home directory"))?;
        let mut buf = home;
        if !stripped.is_empty() {
            buf.push(stripped);
        }
        return Ok(path_to_string(&buf));
    }

    Ok(raw.to_string())
}

/// Return the user's home directory if it can be determined.
pub fn home_directory() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Ok(user_profile) = env::var("USERPROFILE") {
            return Some(PathBuf::from(user_profile));
        }
        if let (Ok(drive), Ok(path)) = (env::var("HOMEDRIVE"), env::var("HOMEPATH")) {
            return Some(PathBuf::from(format!("{drive}{path}")));
        }
        None
    }
    #[cfg(not(windows))]
    {
        env::var("HOME").map(PathBuf::from).ok()
    }
}

/// Convert a path into an owned string using lossless conversion semantics.
pub fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Return `true` when the text contains glob-style wildcards (`*` or `?`).
pub fn contains_wildcards(text: &str) -> bool {
    text.contains('*') || text.contains('?')
}

/// Sort entries in-place using MATLAB-compatible ordering.
pub fn sort_entries(entries: &mut [String]) {
    entries.sort_by(|a, b| compare_names(a, b));
}

/// Compare two file names using case-insensitive ordering on Windows and
/// case-sensitive ordering elsewhere, matching MATLAB's behaviour.
pub fn compare_names(a: &str, b: &str) -> Ordering {
    #[cfg(windows)]
    {
        let lower_a = a.to_ascii_lowercase();
        let lower_b = b.to_ascii_lowercase();
        match lower_a.cmp(&lower_b) {
            Ordering::Equal => a.cmp(b),
            other => other,
        }
    }
    #[cfg(not(windows))]
    {
        a.cmp(b)
    }
}
