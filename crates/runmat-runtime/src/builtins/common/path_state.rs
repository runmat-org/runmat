//! MATLAB-style source search-path state shared by the `path` builtin and
//! filesystem helpers such as `exist` or `which`.
//!
//! The MATLAB search path is represented as a single platform-specific string
//! using the same separator rules that MathWorks MATLAB applies (`;` on
//! Windows, `:` everywhere else).  RunMat keeps the current working directory
//! separate from this list so callers can freely replace or manipulate the path
//! without losing the implicit `pwd` entry that MATLAB always prioritises.
//! Active runtime sessions provide their own [`SearchPath`]; the process state
//! below remains as the initialization and standalone-builtin compatibility
//! context.

use once_cell::sync::Lazy;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, RwLock,
};

use crate::builtins::common::env as runtime_env;

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

/// Ordered, session-owned source search path used by filesystem discovery and
/// runtime callable resolution.
#[derive(Debug)]
pub struct SearchPath {
    current: RwLock<String>,
    generation: AtomicU64,
}

impl SearchPath {
    pub fn new(current: String) -> Self {
        Self {
            current: RwLock::new(current),
            generation: AtomicU64::new(0),
        }
    }

    pub fn current_string(&self) -> String {
        self.current
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_else(|poison| poison.into_inner().clone())
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn replace(&self, new_path: &str) {
        let mut guard = self
            .current
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if *guard != new_path {
            *guard = new_path.to_string();
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn append(&self, segments: &[String]) {
        if segments.is_empty() {
            return;
        }
        let mut guard = self
            .current
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut parts = split_segments(&guard);
        parts.extend(segments.iter().cloned());
        let next = join_parts(&parts);
        if *guard != next {
            *guard = next;
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn initial_path_string() -> String {
    let mut parts = Vec::<String>::new();
    for var in ["RUNMAT_PATH", "MATLABPATH"] {
        if let Ok(value) = runtime_env::var(var) {
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
    if let Some(search_path) = active_search_path() {
        return search_path.current_string();
    }
    PATH_STATE
        .read()
        .map(|guard| guard.current.clone())
        .unwrap_or_else(|poison| poison.into_inner().current.clone())
}

pub fn append_to_path(segments: &[String]) {
    if let Some(search_path) = active_search_path() {
        search_path.append(segments);
        return;
    }
    if segments.is_empty() {
        return;
    }
    let mut guard = PATH_STATE
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut parts = split_segments(&guard.current);
    parts.extend(segments.iter().cloned());
    guard.current = join_parts(&parts);
}

/// Replace the MATLAB path string for the active session. Without an active
/// runtime context this updates the compatibility process state and mirrors it
/// to `RUNMAT_PATH`.
pub fn set_path_string(new_path: &str) {
    if let Some(search_path) = active_search_path() {
        search_path.replace(new_path);
        return;
    }
    if new_path.is_empty() {
        runtime_env::remove_var("RUNMAT_PATH");
    } else {
        runtime_env::set_var("RUNMAT_PATH", new_path);
    }

    let mut guard = PATH_STATE
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.current = new_path.to_string();
}

fn active_search_path() -> Option<Arc<SearchPath>> {
    crate::context::legacy::active().and_then(|context| context.search_path().map(Arc::clone))
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn join_and_split_round_trip() {
        let parts = vec!["/tmp/a".to_string(), "/tmp/b".to_string()];
        let joined = join_parts(&parts);
        assert_eq!(split_segments(&joined), parts);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn active_runtime_context_owns_path_and_generation() {
        let first = Arc::new(SearchPath::new("first".to_string()));
        let first_context = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_search_path(Arc::clone(&first));
        let _first_context = crate::context::RuntimeContextGuard::enter(first_context);
        assert_eq!(current_path_string(), "first");
        set_path_string("first");
        assert_eq!(first.generation(), 0, "no-op replacement is not a mutation");
        let updated = format!("first{PATH_LIST_SEPARATOR}added");
        set_path_string(&updated);
        assert_eq!(current_path_string(), updated);
        assert_eq!(first.generation(), 1);

        let second = Arc::new(SearchPath::new("second".to_string()));
        {
            let second_context = crate::context::RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ))
            .with_search_path(second);
            let _second_context = crate::context::RuntimeContextGuard::enter(second_context);
            assert_eq!(current_path_string(), "second");
        }
        assert_eq!(current_path_string(), updated);
    }
}
