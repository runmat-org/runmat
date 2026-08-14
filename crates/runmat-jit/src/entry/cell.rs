use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use runmat_execution::Digest;
use runmat_types::ProgramFunctionId;

use crate::{invalidation::DependencySnapshot, GenericExecutor};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EntryKey(pub String);

#[derive(Clone)]
pub struct PublishedEntry {
    pub executor: Rc<GenericExecutor>,
    pub entrypoint: ProgramFunctionId,
    pub dependencies: DependencySnapshot,
    pub publication: u64,
    pub tier: PublishedTier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublishedTier {
    Generic,
    Specialized { profile: Digest },
}

#[derive(Default)]
struct EntryState {
    publication: u64,
    generic: Option<PublishedEntry>,
    specialized: BTreeMap<Digest, PublishedEntry>,
}

/// Stable indirection retained while compiled targets are replaced or retired.
#[derive(Clone)]
pub struct EntryCell {
    key: EntryKey,
    state: Rc<RefCell<EntryState>>,
}

impl EntryCell {
    pub(crate) fn new(key: EntryKey) -> Self {
        Self {
            key,
            state: Rc::new(RefCell::new(EntryState::default())),
        }
    }

    pub fn key(&self) -> &EntryKey {
        &self.key
    }

    pub fn resolve(&self, current: &DependencySnapshot) -> Option<PublishedEntry> {
        self.state
            .borrow()
            .generic
            .as_ref()
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .cloned()
    }

    pub(crate) fn publish(
        &self,
        executor: Rc<GenericExecutor>,
        entrypoint: ProgramFunctionId,
        dependencies: DependencySnapshot,
    ) -> Result<PublishedEntry, &'static str> {
        if dependencies.is_empty() {
            return Err("published native entry must declare dependencies");
        }
        let mut state = self.state.borrow_mut();
        state.publication = state
            .publication
            .checked_add(1)
            .ok_or("entry publication generation exhausted")?;
        let target = PublishedEntry {
            executor,
            entrypoint,
            dependencies,
            publication: state.publication,
            tier: PublishedTier::Generic,
        };
        state.generic = Some(target.clone());
        Ok(target)
    }

    pub(crate) fn publish_specialized(
        &self,
        profile: Digest,
        executor: Rc<GenericExecutor>,
        entrypoint: ProgramFunctionId,
        dependencies: DependencySnapshot,
    ) -> Result<PublishedEntry, &'static str> {
        if dependencies.is_empty() {
            return Err("published native entry must declare dependencies");
        }
        let mut state = self.state.borrow_mut();
        state.publication = state
            .publication
            .checked_add(1)
            .ok_or("entry publication generation exhausted")?;
        let target = PublishedEntry {
            executor,
            entrypoint,
            dependencies,
            publication: state.publication,
            tier: PublishedTier::Specialized { profile },
        };
        state.specialized.insert(profile, target.clone());
        Ok(target)
    }

    pub fn resolve_specialized(
        &self,
        profile: Digest,
        current: &DependencySnapshot,
    ) -> Option<PublishedEntry> {
        self.state
            .borrow()
            .specialized
            .get(&profile)
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .cloned()
    }

    pub(crate) fn invalidate_if_changed(&self, current: &DependencySnapshot) -> bool {
        let mut state = self.state.borrow_mut();
        let generic_invalid = state
            .generic
            .as_ref()
            .is_some_and(|target| !target.dependencies.is_satisfied_by(current));
        if generic_invalid {
            state.generic = None;
        }
        let specialized_before = state.specialized.len();
        state
            .specialized
            .retain(|_, target| target.dependencies.is_satisfied_by(current));
        generic_invalid || specialized_before != state.specialized.len()
    }

    pub fn is_published(&self) -> bool {
        self.state.borrow().generic.is_some()
    }

    pub(crate) fn retained_versions(&self, current: &DependencySnapshot) -> usize {
        let state = self.state.borrow();
        usize::from(
            state
                .generic
                .as_ref()
                .is_some_and(|target| target.dependencies.is_satisfied_by(current)),
        ) + state
            .specialized
            .values()
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .count()
    }

    pub(crate) fn retained_code_bytes(&self, current: &DependencySnapshot) -> u64 {
        let state = self.state.borrow();
        state
            .generic
            .iter()
            .chain(state.specialized.values())
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .fold(0_u64, |total, target| {
                total.saturating_add(target.executor.retained_code_bytes())
            })
    }

    pub(crate) fn specialized_profiles(&self, current: &DependencySnapshot) -> Vec<Digest> {
        self.state
            .borrow()
            .specialized
            .iter()
            .filter_map(|(profile, target)| {
                target
                    .dependencies
                    .is_satisfied_by(current)
                    .then_some(*profile)
            })
            .collect()
    }

    pub(crate) fn specialized_version_count(&self, current: &DependencySnapshot) -> usize {
        self.state
            .borrow()
            .specialized
            .values()
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .count()
    }

    pub(crate) fn retire_oldest_specialized(&self) -> Option<(u64, u64)> {
        let mut state = self.state.borrow_mut();
        let (profile, publication, bytes) = state
            .specialized
            .iter()
            .min_by_key(|(_, target)| target.publication)
            .map(|(profile, target)| {
                (
                    *profile,
                    target.publication,
                    target.executor.retained_code_bytes(),
                )
            })?;
        state.specialized.remove(&profile);
        Some((publication, bytes))
    }

    pub(crate) fn oldest_specialized_publication(&self) -> Option<u64> {
        self.state
            .borrow()
            .specialized
            .values()
            .map(|target| target.publication)
            .min()
    }
}
