//! Session-wide MATLAB path state shared between the `path` builtin and
//! filesystem helpers such as `exist` or `which`.
//!
//! The MATLAB search path is represented as a single platform-specific string
//! using the same separator rules that MathWorks MATLAB applies (`;` on
//! Windows, `:` everywhere else).  RunMat keeps the current working directory
//! separate from this list so callers can freely replace or manipulate the path
//! without losing the implicit `pwd` entry that MATLAB always prioritises.

use once_cell::sync::Lazy;
use std::env;
use std::sync::RwLock;

/// Platform-specific separator used when joining MATLAB path entries.
pub const PATH_LIST_SEPARATOR: char = if cfg!(windows) { ';' } else { ':' };

#[derive(Debug, Clone)]
struct PathState {
    /// Current MATLAB path string, excluding the implicit current directory.
    current: String,
}

impl PathState {
    fn initialise() -> Self {
        Self {
            current: initial_path_string(),
        }
    }
}

fn initial_path_string() -> String {
    let mut parts = Vec::<String>::new();
    for var in ["RUNMAT_PATH", "MATLABPATH"] {
        if let Ok(value) = env::var(var) {
            parts.extend(
                value
                    .split(PATH_LIST_SEPARATOR)
                    .map(|part| part.trim())
                    .filter(|part| !part.is_empty())
                    .map(|part| part.to_string()),
            );
        }
    }
    join_parts(&parts)
}

fn join_parts(parts: &[String]) -> String {
    let mut joined = String::new();
    for (idx, part) in parts.iter().enumerate() {
        if idx > 0 {
            joined.push(PATH_LIST_SEPARATOR);
        }
        joined.push_str(part);
    }
    joined
}

static PATH_STATE: Lazy<RwLock<PathState>> = Lazy::new(|| RwLock::new(PathState::initialise()));

/// Return the current MATLAB path string (without the implicit current
/// directory entry).
pub fn current_path_string() -> String {
    PATH_STATE
        .read()
        .map(|guard| guard.current.clone())
        .unwrap_or_else(|poison| poison.into_inner().current.clone())
}

/// Replace the MATLAB path string for the current session. When `new_path` is
/// empty the session path becomes empty and the `RUNMAT_PATH` environment
/// variable is removed.
pub fn set_path_string(new_path: &str) {
    if new_path.is_empty() {
        env::remove_var("RUNMAT_PATH");
    } else {
        env::set_var("RUNMAT_PATH", new_path);
    }

    let mut guard = PATH_STATE
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.current = new_path.to_string();
}

/// Split the current MATLAB path string into individual entries, omitting
/// empty segments and trimming surrounding whitespace.
pub fn current_path_segments() -> Vec<String> {
    let path = current_path_string();
    split_segments(&path)
}

fn split_segments(path: &str) -> Vec<String> {
    path.split(PATH_LIST_SEPARATOR)
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn join_and_split_round_trip() {
        let parts = vec!["/tmp/a".to_string(), "/tmp/b".to_string()];
        let joined = join_parts(&parts);
        assert_eq!(split_segments(&joined), parts);
    }
}
