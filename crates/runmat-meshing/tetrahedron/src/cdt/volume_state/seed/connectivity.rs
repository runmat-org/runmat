//! Compare canonical connectivity from local symmetric differences. Common
//! baseline cells need only an indexed successor query, never a full inventory.
use super::*;
use std::cmp::Ordering;
use std::ops::Bound::{Excluded, Unbounded};
type CellKey = [StableDigest; 4];

pub(super) struct ConnectivityDelta {
    pub(super) removed: BTreeSet<CellKey>,
    pub(super) created: BTreeSet<CellKey>,
}
impl ConnectivityDelta {
    pub(super) fn from_edits<'a>(edits: impl IntoIterator<Item = &'a TopologyEdit>) -> Self {
        let mut result = Self {
            removed: BTreeSet::new(),
            created: BTreeSet::new(),
        };
        for edit in edits {
            match edit {
                TopologyEdit::ReplaceCavity { removed, created }
                | TopologyEdit::RelocateCavity {
                    removed, created, ..
                } => {
                    for key in removed.iter().copied() {
                        result.remove(key);
                    }
                    for (key, _) in created {
                        result.insert(*key);
                    }
                }
                TopologyEdit::ReplaceNode {
                    source_identity,
                    replacement,
                    affected,
                } => {
                    for key in affected {
                        result.remove(*key);
                    }
                    for key in affected {
                        result.insert(key.map(|identity| {
                            if identity == *source_identity {
                                replacement.identity
                            } else {
                                identity
                            }
                        }));
                    }
                }
                TopologyEdit::AppendNode(_) => {}
            }
        }
        result
    }
    fn remove(&mut self, mut key: CellKey) {
        key.sort_unstable();
        if !self.created.remove(&key) {
            self.removed.insert(key);
        }
    }
    fn insert(&mut self, mut key: CellKey) {
        key.sort_unstable();
        if !self.removed.remove(&key) {
            self.created.insert(key);
        }
    }
    fn contains(&self, key: &CellKey, baseline: &WorkingVolumeQuality) -> bool {
        self.created.contains(key)
            || (!self.removed.contains(key) && baseline.baseline_contains(key))
    }
    fn successor(&self, key: &CellKey, baseline: &WorkingVolumeQuality) -> Option<CellKey> {
        let mut original = baseline.baseline_successor(key);
        while original.is_some_and(|key| self.removed.contains(&key)) {
            original = baseline.baseline_successor(&original.unwrap());
        }
        let created = self
            .created
            .range((Excluded(*key), Unbounded))
            .next()
            .copied();
        match (original, created) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    pub(super) fn compare(&self, other: &Self, baseline: &WorkingVolumeQuality) -> Ordering {
        let mut changed = self
            .removed
            .union(&self.created)
            .copied()
            .collect::<BTreeSet<_>>();
        changed.extend(other.removed.union(&other.created).copied());
        for key in changed {
            match (
                self.contains(&key, baseline),
                other.contains(&key, baseline),
            ) {
                (true, false) => {
                    return if other.successor(&key, baseline).is_some() {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                (false, true) => {
                    return if self.successor(&key, baseline).is_some() {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                _ => {}
            }
        }
        Ordering::Equal
    }
}
