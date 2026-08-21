use std::collections::BTreeMap;

/// Stable identity for one independently invalidatable native-code input.
///
/// The identities are executor-neutral and available on WASM. Only the native
/// entry registry consumes them to retain machine code.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DependencyKey {
    Program(String),
    Catalog(String),
    SessionFunction(String),
    SessionCatalog,
    Provider(u32),
}

/// Monotonic generation within one JIT/session dependency tracker.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DependencyGeneration(u64);

impl DependencyGeneration {
    pub const INITIAL: Self = Self(1);

    pub fn get(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// Exact dependency generations used to compile one published entry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DependencySnapshot {
    generations: BTreeMap<DependencyKey, DependencyGeneration>,
}

impl DependencySnapshot {
    pub fn insert(&mut self, key: DependencyKey, generation: DependencyGeneration) {
        self.generations.insert(key, generation);
    }

    pub fn generation(&self, key: &DependencyKey) -> Option<DependencyGeneration> {
        self.generations.get(key).copied()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&DependencyKey, DependencyGeneration)> {
        self.generations
            .iter()
            .map(|(key, generation)| (key, *generation))
    }

    pub fn is_empty(&self) -> bool {
        self.generations.is_empty()
    }

    /// Whether `current` contains the same generation for every dependency in
    /// this compilation snapshot. Unrelated current dependencies do not make a
    /// target stale.
    pub fn is_satisfied_by(&self, current: &Self) -> bool {
        self.iter()
            .all(|(key, generation)| current.generation(key) == Some(generation))
    }
}
