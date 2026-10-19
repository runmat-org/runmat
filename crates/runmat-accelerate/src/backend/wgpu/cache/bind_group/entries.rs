//! The existing cache table and reverse index share one mutation boundary.
use super::{BindGroupKey, CacheInvalidationError};
use std::collections::{hash_map::Entry, HashMap, HashSet};

pub(super) struct BindGroupEntries<T> {
    pub(super) groups: HashMap<BindGroupKey, T>,
    by_buffer: HashMap<usize, HashSet<BindGroupKey>>,
}

impl<T> Default for BindGroupEntries<T> {
    fn default() -> Self {
        Self {
            groups: HashMap::new(),
            by_buffer: HashMap::new(),
        }
    }
}

impl<T> BindGroupEntries<T> {
    /// Return replaced resources to the caller; never destroy them under the
    /// cache lock. All index work precedes replacement of the resource itself.
    pub(super) fn insert(&mut self, key: BindGroupKey, value: T) -> Option<T> {
        for binding in &key.bindings {
            self.by_buffer
                .entry(binding.buffer_ptr)
                .or_default()
                .insert(key.clone());
        }
        self.groups.insert(key, value)
    }

    pub(super) fn remove_buffer(
        &mut self,
        buffer_ptr: usize,
    ) -> Result<Vec<T>, CacheInvalidationError> {
        let count = self.by_buffer.get(&buffer_ptr).map_or(0, HashSet::len);
        let mut removed = Vec::new();
        // No mutation occurs if storage for the removed resources is unavailable.
        removed
            .try_reserve_exact(count)
            .map_err(|_| CacheInvalidationError::BookkeepingCapacity)?;
        let Some(keys) = self.by_buffer.remove(&buffer_ptr) else {
            return Ok(removed);
        };
        for key in keys {
            if let Some(resource) = self.groups.remove(&key) {
                removed.push(resource);
            }
            for binding in &key.bindings {
                if binding.buffer_ptr == buffer_ptr {
                    continue;
                }
                if let Entry::Occupied(mut entry) = self.by_buffer.entry(binding.buffer_ptr) {
                    entry.get_mut().remove(&key);
                    if entry.get().is_empty() {
                        entry.remove();
                    }
                }
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
#[path = "entries_tests.rs"]
mod tests;
