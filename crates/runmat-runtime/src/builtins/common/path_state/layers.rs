//! Provenance for the ordered search path. Presentation remains a path-list
//! string, but userpath mutations must not erase an unrelated equal spelling.

use super::{join_parts, split_segments};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UserAuthorityState {
    /// An embedding installed only an explicit local search path.
    Unscoped,
    Current,
    /// A durable mutation may have committed without local publication.
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathOrigin {
    User,
    Environment,
    Installation,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PathEntry {
    pub(crate) path: String,
    pub(crate) origin: PathOrigin,
}

impl PathEntry {
    pub(crate) fn new(path: String, origin: PathOrigin) -> Self {
        Self { path, origin }
    }
}

#[derive(Debug)]
pub(super) struct SessionPaths {
    pub(super) current: Vec<PathEntry>,
    pub(super) factory: Vec<PathEntry>,
    pub(super) user_revision: Option<u64>,
    pub(super) user_authority: UserAuthorityState,
}

impl SessionPaths {
    pub(super) fn from_flat(path: &str) -> Self {
        let entries = split_segments(path)
            .into_iter()
            .map(|path| PathEntry::new(path, PathOrigin::Manual))
            .collect::<Vec<_>>();
        Self {
            current: entries.clone(),
            factory: entries,
            user_revision: None,
            user_authority: UserAuthorityState::Unscoped,
        }
    }

    pub(super) fn current_string(&self) -> String {
        join_parts(
            &self
                .current
                .iter()
                .map(|entry| entry.path.clone())
                .collect::<Vec<_>>(),
        )
    }

    pub(super) fn replace_flat(&mut self, path: &str) -> bool {
        let next = Self::from_flat(path).current;
        if self.current == next {
            return false;
        }
        self.current = next;
        true
    }

    pub(super) fn append(&mut self, paths: &[String]) -> bool {
        if paths.is_empty() {
            return false;
        }
        self.current.extend(
            paths
                .iter()
                .cloned()
                .map(|path| PathEntry::new(path, PathOrigin::Manual)),
        );
        true
    }

    pub(super) fn replace_user(&mut self, path: Option<String>) -> bool {
        let before = self.current.clone();
        self.current
            .retain(|entry| entry.origin != PathOrigin::User);
        self.factory
            .retain(|entry| entry.origin != PathOrigin::User);
        if let Some(path) = path.filter(|path| !path.is_empty()) {
            let entry = PathEntry::new(path, PathOrigin::User);
            self.current.insert(0, entry.clone());
            self.factory.insert(0, entry);
        }
        self.current != before
    }
}
