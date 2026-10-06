//! Session-lane ownership for admitted dynamic source products.
//!
//! Readers receive owned snapshots, never a guard held across suspension.
//! Replaced products and bindings retire after mutable access is released,
//! allowing their context/GC owners to reenter the session during destruction.

use super::DynamicFunctionCacheEntry;
use runmat_value::BoundFunctionBinding;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub(super) struct DynamicFunctionCache {
    entries: RefCell<HashMap<PathBuf, DynamicFunctionCacheEntry>>,
}

impl DynamicFunctionCache {
    pub(super) fn get(&self, path: &Path) -> Option<DynamicFunctionCacheEntry> {
        self.entries.borrow().get(path).cloned()
    }

    pub(super) fn insert(&self, path: PathBuf, entry: DynamicFunctionCacheEntry) {
        let retired = self.entries.borrow_mut().insert(path, entry);
        drop(retired);
    }

    pub(super) fn replace_binding(&self, path: &Path, binding: BoundFunctionBinding) {
        let retired = {
            let mut entries = self.entries.borrow_mut();
            if let Some(entry) = entries.get_mut(path) {
                std::mem::replace(&mut entry.binding, binding)
            } else {
                binding
            }
        };
        drop(retired);
    }

    pub(super) fn clear(&self) {
        let retired = std::mem::take(&mut *self.entries.borrow_mut());
        drop(retired);
    }

    pub(super) fn clear_named(&self, name: &str) {
        let retired = {
            let mut entries = self.entries.borrow_mut();
            let paths = entries
                .keys()
                .filter(|path| super::dynamic::path_matches_clear_name(path, name))
                .cloned()
                .collect::<Vec<_>>();
            let mut retired = Vec::with_capacity(paths.len());
            for path in paths {
                if let Some(entry) = entries.remove(&path) {
                    retired.push(entry);
                }
            }
            retired
        };
        drop(retired);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> HashMap<PathBuf, DynamicFunctionCacheEntry> {
        self.entries.borrow().clone()
    }
}
