use std::sync::Arc;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use runmat_execution::Digest;
use runmat_types::FunctionId;

use crate::invalidation::DependencySnapshot;
use runmat_native_executor::NativeExecutor;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EntryKey(pub String);

impl std::borrow::Borrow<str> for EntryKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Clone)]
pub struct PublishedEntry {
    pub executor: Rc<NativeExecutor>,
    /// Local compiled coordinate in this exact executor, not a portable owner.
    pub entrypoint: FunctionId,
    /// Frozen publication authority. Cloning a ready product must not copy
    /// its dependency map or create a second mutable snapshot.
    pub dependencies: Arc<DependencySnapshot>,
    pub publication: u64,
    pub tier: PublishedTier,
    /// Complete Core preparation, lowering, code generation, and binding time
    /// for this exact published product. Ready execution never pays it again.
    pub compilation: NativeCompilationEvidence,
}

/// Metadata for the last budget-retired specialization in this entry. It
/// cannot execute code or restore a publication; it can only price preparing
/// the identical recipe again while its dependency generations remain current.
#[derive(Clone)]
pub struct NativePreparationQuote {
    pub entrypoint: FunctionId,
    pub dependencies: Arc<DependencySnapshot>,
    pub publication: u64,
    pub tier: PublishedTier,
    pub compilation: NativeCompilationEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCompilationEvidence {
    pub prepare_ns: u64,
    pub lower_codegen_bind_ns: u64,
}

impl NativeCompilationEvidence {
    pub fn total_ns(self) -> Option<u64> {
        self.prepare_ns.checked_add(self.lower_codegen_bind_ns)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublishedTier {
    Generic,
    Specialized { profile: Digest },
}

/// Publication identities observed for one entry before tiering may prepare
/// another product. This snapshot does not authenticate dependencies or pin
/// executable code; resolution must still validate the selected publication.
#[derive(Clone, Copy, Default)]
pub struct EntryPublications {
    generic: Option<u64>,
    specialized: Option<(Digest, u64)>,
}

impl EntryPublications {
    /// Compare a subsequently resolved product from the same registry entry.
    pub fn contains(self, entry: &PublishedEntry) -> bool {
        match entry.tier {
            PublishedTier::Generic => self.generic == Some(entry.publication),
            PublishedTier::Specialized { profile } => {
                self.specialized == Some((profile, entry.publication))
            }
        }
    }
}

#[derive(Default)]
struct EntryState {
    publication: u64,
    generic: Option<PublishedEntry>,
    specialized: BTreeMap<Digest, PublishedEntry>,
    retired_preparation: Option<NativePreparationQuote>,
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

    pub(crate) fn publications(&self, profile: Option<Digest>) -> EntryPublications {
        let state = self.state.borrow();
        EntryPublications {
            generic: state.generic.as_ref().map(|entry| entry.publication),
            specialized: profile.and_then(|profile| {
                state
                    .specialized
                    .get(&profile)
                    .map(|entry| (profile, entry.publication))
            }),
        }
    }

    pub fn resolve(&self, current: &DependencySnapshot) -> Option<PublishedEntry> {
        self.state
            .borrow()
            .generic
            .as_ref()
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .cloned()
    }

    pub(crate) fn resolve_current(
        &self,
        current: &crate::invalidation::DependencyTracker,
        profile: Option<runmat_execution::Digest>,
    ) -> Option<PublishedEntry> {
        let state = self.state.borrow();
        let entry = match profile {
            Some(profile) => state.specialized.get(&profile),
            None => state.generic.as_ref(),
        }?;
        let dependencies_current = entry
            .dependencies
            .iter()
            .all(|(key, generation)| current.generation(key) == Some(generation));
        dependencies_current.then(|| entry.clone())
    }

    pub(crate) fn publish(
        &self,
        executor: Rc<NativeExecutor>,
        entrypoint: FunctionId,
        dependencies: DependencySnapshot,
        compilation: NativeCompilationEvidence,
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
            dependencies: Arc::new(dependencies),
            publication: state.publication,
            tier: PublishedTier::Generic,
            compilation,
        };
        state.generic = Some(target.clone());
        Ok(target)
    }

    pub(crate) fn publish_specialized(
        &self,
        profile: Digest,
        executor: Rc<NativeExecutor>,
        entrypoint: FunctionId,
        dependencies: DependencySnapshot,
        compilation: NativeCompilationEvidence,
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
            dependencies: Arc::new(dependencies),
            publication: state.publication,
            tier: PublishedTier::Specialized { profile },
            compilation,
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
        if state
            .retired_preparation
            .as_ref()
            .is_some_and(|quote| !quote.dependencies.is_satisfied_by(current))
        {
            state.retired_preparation = None;
        }
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
        let retired = state.specialized.remove(&profile)?;
        state.retired_preparation = Some(NativePreparationQuote {
            entrypoint: retired.entrypoint,
            dependencies: retired.dependencies,
            publication: retired.publication,
            tier: retired.tier,
            compilation: retired.compilation,
        });
        Some((publication, bytes))
    }

    pub(crate) fn preparation_quote(
        &self,
        profile: Digest,
        current: &DependencySnapshot,
    ) -> Option<NativePreparationQuote> {
        self.state
            .borrow()
            .retired_preparation
            .as_ref()
            .filter(|quote| {
                quote.tier == (PublishedTier::Specialized { profile })
                    && quote.dependencies.is_satisfied_by(current)
            })
            .cloned()
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
