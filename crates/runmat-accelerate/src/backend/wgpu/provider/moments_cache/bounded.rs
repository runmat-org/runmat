use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

use super::CacheBudget;
use runmat_accelerate_api::allocation::WeakGpuTensorHandle;
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

struct Pair<H, A> {
    mean: H,
    ex2: H,
    source_view: Option<WeakGpuTensorHandle>,
    allocations: [A; 3],
    bytes: u64,
    // Fund variable request-key payload until remove() has cleared every
    // reverse index and LRU key. This is separate from GPU allocation bytes.
    _metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

/// One already-proven physical moment pair and its exact cache ownership.
/// Grouping the fields makes insertion consume a single coherent entry.
pub(super) struct MomentEntry<K, H, A> {
    pub key: K,
    pub allocations: [A; 3],
    pub source_view: Option<WeakGpuTensorHandle>,
    pub mean: H,
    pub ex2: H,
    pub bytes: u64,
    pub metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

/// `K` identifies one exact source view and reduction-dimension sequence;
/// `A` is the canonical physical allocation identity, never a view ordinal.
pub(super) struct BoundedMoments<K, H, A> {
    budget: CacheBudget,
    entries: HashMap<K, Pair<H, A>>,
    oldest_first: VecDeque<K>,
    bytes: u64,
}

impl<K, H, A> BoundedMoments<K, H, A>
where
    K: Clone + Eq + Hash,
    H: Clone,
    A: Clone + Eq + Hash,
{
    pub(super) fn entry_inline_bytes() -> usize {
        std::mem::size_of::<K>() + std::mem::size_of::<Pair<H, A>>()
    }
    pub(super) fn try_new(budget: CacheBudget) -> Result<Self, std::collections::TryReserveError> {
        let mut entries = HashMap::new();
        let entries_bound = if budget.max_bytes == 0 {
            0
        } else {
            budget.max_entries
        };
        entries.try_reserve(entries_bound)?;
        let mut oldest_first = VecDeque::new();
        oldest_first.try_reserve(entries_bound)?;
        Ok(Self {
            budget,
            // The provider-wide index backing is allocated at provider
            // construction, before any selected request can insert a key.
            entries,
            oldest_first,
            bytes: 0,
        })
    }

    #[cfg(test)]
    pub(super) fn new(budget: CacheBudget) -> Self {
        Self::try_new(budget).expect("test cache capacity")
    }

    pub(super) fn get(&mut self, key: &K) -> Option<(H, H)> {
        let pair = self.entries.get(key)?;
        let result = (pair.mean.clone(), pair.ex2.clone());
        self.oldest_first.retain(|candidate| candidate != key);
        self.oldest_first.push_back(key.clone());
        Some(result)
    }

    /// Returns all strong handles detached by replacement/eviction/bypass.
    /// The owner must drop them after releasing its mutex.
    pub(super) fn insert(&mut self, entry: MomentEntry<K, H, A>) -> (bool, Vec<(H, H)>) {
        let MomentEntry {
            key,
            allocations,
            source_view,
            mean,
            ex2,
            bytes,
            metadata_claim,
        } = entry;
        let mut detached = Vec::new();
        if !self.budget.admits(1, bytes) {
            detached.push((mean, ex2));
            return (false, detached);
        }
        if let Some(pair) = self.remove(&key) {
            detached.push((pair.mean, pair.ex2));
        }
        while self.entries.len() >= self.budget.max_entries
            || self
                .bytes
                .checked_add(bytes)
                .is_none_or(|total| total > self.budget.max_bytes)
        {
            let Some(oldest) = self.oldest_first.front().cloned() else {
                break;
            };
            if let Some(pair) = self.remove(&oldest) {
                detached.push((pair.mean, pair.ex2));
            }
        }
        // Three physical identities are inline in Pair. Retirement scans the
        // bounded provider index instead of lazily growing reverse hash sets.
        self.entries.insert(
            key.clone(),
            Pair {
                mean,
                ex2,
                source_view,
                allocations,
                bytes,
                _metadata_claim: metadata_claim,
            },
        );
        self.bytes += bytes;
        self.oldest_first.push_back(key);
        (true, detached)
    }

    /// Detaches all aliases and dimension variants for this physical record.
    pub(super) fn retire(&mut self, allocation: &A) -> Vec<(H, H)> {
        let mut detached = Vec::new();
        while let Some(key) = self
            .oldest_first
            .iter()
            .find(|key| {
                self.entries
                    .get(*key)
                    .is_some_and(|pair| pair.allocations.contains(allocation))
            })
            .cloned()
        {
            if let Some(pair) = self.remove(&key) {
                detached.push((pair.mean, pair.ex2));
            }
        }
        detached
    }

    /// Snapshot weak exact-view eligibility for checking outside the cache lock.
    pub(super) fn source_views(&self) -> Vec<(K, WeakGpuTensorHandle)> {
        self.entries
            .iter()
            .filter_map(|(key, pair)| {
                pair.source_view
                    .as_ref()
                    .map(|source| (key.clone(), source.clone()))
            })
            .collect()
    }

    pub(super) fn remove_keys(&mut self, keys: &[K]) -> Vec<(H, H)> {
        keys.iter()
            .filter_map(|key| self.remove(key))
            .map(|pair| (pair.mean, pair.ex2))
            .collect()
    }

    pub(super) fn detach_all(&mut self) -> Vec<(H, H)> {
        self.oldest_first.clear();
        self.bytes = 0;
        std::mem::take(&mut self.entries)
            .into_values()
            .map(|pair| (pair.mean, pair.ex2))
            .collect()
    }

    fn remove(&mut self, key: &K) -> Option<Pair<H, A>> {
        let pair = self.entries.remove(key)?;
        self.bytes -= pair.bytes;
        self.oldest_first.retain(|candidate| candidate != key);
        Some(pair)
    }

    #[cfg(test)]
    pub(super) fn retained(&self) -> (usize, u64) {
        (self.entries.len(), self.bytes)
    }
}
