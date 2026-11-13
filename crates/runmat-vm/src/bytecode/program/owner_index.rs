//! Derived lookup for opaque definition-owner aliases inside one registry.
//! The current and retired bodies remain the only authority for an owner.

use std::collections::HashMap;
use std::sync::Arc;

use runmat_execution::FunctionBindingOrigin;
use runmat_hir::FunctionId;

use super::{FunctionBytecode, FunctionRegistryData};

#[derive(Clone, Copy, Debug)]
struct OwnerBinding {
    function: FunctionId,
    origin: Option<FunctionBindingOrigin>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct OwnerIndex {
    by_pointer: HashMap<usize, Vec<OwnerBinding>>,
}

impl OwnerIndex {
    pub(super) fn from_bodies<'a>(bodies: impl IntoIterator<Item = &'a FunctionBytecode>) -> Self {
        let mut index = Self::default();
        for body in bodies {
            index.insert(body);
        }
        index
    }

    pub(super) fn conflicts(&self, body: &FunctionBytecode) -> bool {
        self.by_pointer
            .get(&owner_pointer(body))
            .is_some_and(|bindings| {
                bindings.iter().any(|existing| {
                    existing.function != body.function
                        || existing.origin.is_none()
                        || body.binding_origin.is_none()
                        || existing.origin != body.binding_origin
                })
            })
    }

    pub(super) fn insert(&mut self, body: &FunctionBytecode) {
        self.by_pointer
            .entry(owner_pointer(body))
            .or_default()
            .push(OwnerBinding {
                function: body.function,
                origin: body.binding_origin,
            });
    }

    pub(super) fn remove(&mut self, body: &FunctionBytecode) {
        let pointer = owner_pointer(body);
        let Some(bindings) = self.by_pointer.get_mut(&pointer) else {
            return;
        };
        if let Some(position) = bindings.iter().position(|entry| {
            entry.function == body.function && entry.origin == body.binding_origin
        }) {
            bindings.swap_remove(position);
        }
        if bindings.is_empty() {
            self.by_pointer.remove(&pointer);
        }
    }
}

fn owner_pointer(body: &FunctionBytecode) -> usize {
    Arc::as_ptr(&body.definition_owner) as usize
}

impl FunctionRegistryData {
    /// Detach one current body and its source membership. The index is only a
    /// derived lookup; callers decide whether the old body remains retired.
    pub(super) fn detach_definition(&mut self, function: FunctionId) -> Option<FunctionBytecode> {
        let removed = self.functions.remove(&function)?;
        if let Some(source_id) = removed.source_id {
            if let Some(functions) = self.source_functions.get_mut(&source_id) {
                functions.retain(|id| *id != function);
                if functions.is_empty() {
                    self.source_functions.remove(&source_id);
                }
            }
        }
        if let Some(index) = self.owner_index.get_mut().as_mut() {
            Arc::make_mut(index).remove(&removed);
        }
        Some(removed)
    }

    pub(super) fn retain_definition(&mut self, body: FunctionBytecode) {
        if let Some(index) = self.owner_index.get_mut().as_mut() {
            Arc::make_mut(index).insert(&body);
        }
        self.retired.entry(body.function).or_default().push(body);
    }
}
