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

#[path = "path_state/layers.rs"]
mod layers;
pub(crate) use layers::{PathEntry, PathOrigin};
#[path = "path_state/user_setting.rs"]
mod user_setting;
pub use user_setting::{CapturedUserPathAuthority, ResolvedUserPath};

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
    paths: RwLock<layers::SessionPaths>,
    generation: AtomicU64,
}

impl SearchPath {
    pub fn new(current: String) -> Self {
        Self {
            paths: RwLock::new(layers::SessionPaths::from_flat(&current)),
            generation: AtomicU64::new(0),
        }
    }

    pub fn current_string(&self) -> String {
        self.paths
            .read()
            .map(|guard| guard.current_string())
            .unwrap_or_else(|poison| poison.into_inner().current_string())
    }

    pub(crate) fn snapshot(&self) -> Vec<PathEntry> {
        let guard = self
            .paths
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.current.clone()
    }

    fn install_factory_at_revision(&self, entries: Vec<PathEntry>, revision: u64) -> bool {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard
            .user_revision
            .is_some_and(|current| revision < current)
        {
            return false;
        }
        let before = guard.current_string();
        let authority_changed = guard.user_revision != Some(revision)
            || guard.user_authority != layers::UserAuthorityState::Current;
        let retained_manual = guard
            .current
            .iter()
            .filter(|entry| entry.origin == PathOrigin::Manual)
            .cloned()
            .collect::<Vec<_>>();
        guard.factory = entries.clone();
        guard.current = retained_manual.into_iter().chain(entries).collect();
        guard.user_revision = Some(revision);
        guard.user_authority = layers::UserAuthorityState::Current;
        if guard.current_string() != before || authority_changed {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        true
    }

    /// Install the session's captured default layers after its application
    /// user-path setting has been read. This must precede source resolution.
    pub fn install_captured_factory(
        &self,
        user: ResolvedUserPath,
        user_revision: u64,
        environment: &str,
        installation: Option<String>,
    ) -> bool {
        let mut entries = Vec::new();
        if let Some(path) = user.search_segment {
            entries.push(PathEntry::new(path, PathOrigin::User));
        }
        entries.extend(
            split_segments(environment)
                .into_iter()
                .map(|path| PathEntry::new(path, PathOrigin::Environment)),
        );
        if let Some(path) = installation {
            entries.push(PathEntry::new(path, PathOrigin::Installation));
        }
        self.install_factory_at_revision(entries, user_revision)
    }

    pub fn user_authority_current(&self) -> bool {
        self.paths
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .user_authority
            == layers::UserAuthorityState::Current
    }

    fn source_authority_current(&self) -> bool {
        let guard = self
            .paths
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.user_authority != layers::UserAuthorityState::Stale
    }

    pub fn mark_user_authority_stale(&self) {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.user_authority != layers::UserAuthorityState::Stale {
            guard.user_authority = layers::UserAuthorityState::Stale;
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub fn replace_user_if_newer(&self, revision: u64, path: Option<String>) -> bool {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard
            .user_revision
            .is_some_and(|current| revision <= current)
        {
            return false;
        }
        guard.user_revision = Some(revision);
        guard.replace_user(path);
        guard.user_authority = layers::UserAuthorityState::Current;
        self.generation.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Rebind a stale local path interpretation after an uncertain or
    /// committed-but-unavailable storage outcome. An equal durable revision
    /// may still need a new filesystem/CWD authority.
    pub fn rehydrate_user(&self, revision: u64, path: Option<String>) -> bool {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard
            .user_revision
            .is_some_and(|current| revision < current)
        {
            return false;
        }
        let authority_changed = guard.user_revision != Some(revision)
            || guard.user_authority != layers::UserAuthorityState::Current;
        guard.user_revision = Some(revision);
        if guard.replace_user(path) || authority_changed {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        guard.user_authority = layers::UserAuthorityState::Current;
        true
    }

    pub(crate) fn restore_factory(&self) {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.current != guard.factory {
            guard.current = guard.factory.clone();
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(crate) fn replace_entries(&self, entries: Vec<PathEntry>) {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.current != entries {
            guard.current = entries;
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Invalidate lookup facts even when the textual search path is unchanged.
    pub fn refresh_generation(&self) -> Result<u64, PathGenerationExhausted> {
        self.generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map(|previous| previous + 1)
            .map_err(|_| PathGenerationExhausted)
    }

    fn replace(&self, new_path: &str) {
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.replace_flat(new_path) {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn append(&self, segments: &[String]) {
        if segments.is_empty() {
            return;
        }
        let mut guard = self
            .paths
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.append(segments) {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
    }
}

#[derive(Debug)]
pub struct PathGenerationExhausted;

impl std::fmt::Display for PathGenerationExhausted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("search-path revision space exhausted")
    }
}

impl std::error::Error for PathGenerationExhausted {}

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

/// Capture configured environment path layers without inheriting process-global
/// compatibility mutations from an unrelated session.
pub fn environment_path_string() -> String {
    initial_path_string()
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

/// Snapshot the current entries with provenance. Path-list mutators preserve
/// the origin of retained entries rather than round-tripping through text.
pub(crate) fn current_path_entries() -> Vec<PathEntry> {
    if let Some(search_path) = active_search_path() {
        return search_path.snapshot();
    }
    split_segments(&current_path_string())
        .into_iter()
        .map(|path| PathEntry::new(path, PathOrigin::Manual))
        .collect()
}

pub(crate) fn set_path_entries(entries: Vec<PathEntry>) -> crate::BuiltinResult<()> {
    if let Some(search_path) = active_search_path() {
        search_path.replace_entries(entries);
    } else {
        set_path_string(&join_parts(
            &entries
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
        ))?;
    }
    Ok(())
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
pub fn set_path_string(new_path: &str) -> crate::BuiltinResult<()> {
    if new_path.contains('\0') {
        return Err(crate::runtime_descriptor_error(
            "path",
            &runmat_builtins::PATH_ERROR_INVALID_INPUT,
        ));
    }
    if let Some(search_path) = active_search_path() {
        search_path.replace(new_path);
        return Ok(());
    }
    let mirrored = if new_path.is_empty() {
        runtime_env::try_remove_var("RUNMAT_PATH")
    } else {
        runtime_env::try_set_var("RUNMAT_PATH", new_path)
    };
    mirrored.map_err(|source| {
        let mut error = crate::runtime_descriptor_error(
            "path",
            &runmat_builtins::PATH_ERROR_ENVIRONMENT_UPDATE_FAILED,
        );
        error.source = Some(Box::new(source));
        error
    })?;

    let mut guard = PATH_STATE
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.current = new_path.to_string();
    Ok(())
}

fn active_search_path() -> Option<Arc<SearchPath>> {
    crate::context::legacy::active().and_then(|context| context.search_path().map(Arc::clone))
}

/// Prevent source discovery from consuming an old user entry after a durable
/// commit has an uncertain or locally unpublishable outcome.
pub(crate) fn source_authority_current() -> bool {
    active_search_path().is_none_or(|search| search.source_authority_current())
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
#[path = "path_state/tests.rs"]
pub(crate) mod tests;
