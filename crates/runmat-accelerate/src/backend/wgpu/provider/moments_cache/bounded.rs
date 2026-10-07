use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use super::CacheBudget;
use runmat_accelerate_api::allocation::WeakGpuTensorHandle;

struct Pair<H, A> {
    mean: H,
    ex2: H,
    source_view: Option<WeakGpuTensorHandle>,
    allocations: Vec<A>,
    bytes: u64,
}

/// `K` identifies one exact source view and reduction-dimension sequence;
/// `A` is the canonical physical allocation identity, never a view ordinal.
pub(super) struct BoundedMoments<K, H, A> {
    budget: CacheBudget,
    entries: HashMap<K, Pair<H, A>>,
    oldest_first: VecDeque<K>,
    by_allocation: HashMap<A, HashSet<K>>,
    bytes: u64,
}

impl<K, H, A> BoundedMoments<K, H, A>
where
    K: Clone + Eq + Hash,
    H: Clone,
    A: Clone + Eq + Hash,
{
    pub(super) fn new(budget: CacheBudget) -> Self {
        Self {
            budget,
            entries: HashMap::new(),
            oldest_first: VecDeque::new(),
            by_allocation: HashMap::new(),
            bytes: 0,
        }
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
    pub(super) fn insert(
        &mut self,
        key: K,
        allocations: impl IntoIterator<Item = A>,
        source_view: Option<WeakGpuTensorHandle>,
        mean: H,
        ex2: H,
        bytes: u64,
    ) -> (bool, Vec<(H, H)>) {
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
        let allocations: Vec<A> = allocations
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        for allocation in &allocations {
            self.by_allocation
                .entry(allocation.clone())
                .or_default()
                .insert(key.clone());
        }
        self.entries.insert(
            key.clone(),
            Pair {
                mean,
                ex2,
                source_view,
                allocations,
                bytes,
            },
        );
        self.bytes += bytes;
        self.oldest_first.push_back(key);
        (true, detached)
    }

    /// Detaches all aliases and dimension variants for this physical record.
    pub(super) fn retire(&mut self, allocation: &A) -> Vec<(H, H)> {
        let keys = self
            .by_allocation
            .get(allocation)
            .cloned()
            .unwrap_or_default();
        let mut detached = Vec::with_capacity(keys.len());
        let ordered: Vec<K> = self
            .oldest_first
            .iter()
            .filter(|key| keys.contains(*key))
            .cloned()
            .collect();
        for key in ordered {
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
        self.by_allocation.clear();
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
        for allocation in &pair.allocations {
            let empty = if let Some(keys) = self.by_allocation.get_mut(allocation) {
                keys.remove(key);
                keys.is_empty()
            } else {
                false
            };
            if empty {
                self.by_allocation.remove(allocation);
            }
        }
        Some(pair)
    }

    #[cfg(test)]
    pub(super) fn retained(&self) -> (usize, u64) {
        (self.entries.len(), self.bytes)
    }
}
