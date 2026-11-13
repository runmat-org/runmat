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

    pub(super) fn declares_super_class(
        entry: &DynamicFunctionCacheEntry,
        parent: &runmat_types::ClassIdentity,
    ) -> bool {
        let declares = |instruction: &runmat_vm::Instr| matches!(instruction, runmat_vm::Instr::RegisterClass { name, .. } if name == parent);
        entry.admitted.bytecode().instructions.iter().any(declares)
            || entry.admitted.registry().functions.values().any(|body| {
                body.binding_origin.map(|origin| origin.program)
                    == Some(entry.entrypoint_origin.program)
                    && (body.class_method_owner.as_ref().is_some_and(|owner| {
                        owner.declaring_class == *parent
                            && owner.role == runmat_types::ClassMethodRole::Constructor
                            && !owner.is_static
                    }) || body.instructions.iter().any(declares))
            })
    }

    pub(super) fn insert(&self, path: PathBuf, entry: DynamicFunctionCacheEntry) {
        let retired = self.entries.borrow_mut().insert(path, entry);
        drop(retired);
    }

    pub(super) fn replace_binding(
        &self,
        path: &Path,
        binding: BoundFunctionBinding,
        super_constructors: Vec<runmat_runtime::context::RetainedSuperConstructorBinding>,
        super_dependency_origins: std::collections::HashSet<super::DynamicDependencyIdentity>,
    ) {
        let retired = {
            let mut entries = self.entries.borrow_mut();
            if let Some(entry) = entries.get_mut(path) {
                let old_binding = std::mem::replace(&mut entry.binding, binding);
                let old_super_constructors =
                    std::mem::replace(&mut entry.super_constructors, super_constructors);
                entry.super_dependency_origins = super_dependency_origins;
                (old_binding, old_super_constructors)
            } else {
                (binding, super_constructors)
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
