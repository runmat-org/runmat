//! Persistent session bindings, invalidation leases, and native continuation authority.

use super::{with_state, with_state_mut, SessionVariableAuthority, SessionVariableState};
use runmat_value::Value;
use std::rc::{Rc, Weak};

impl SessionVariableAuthority {
    pub fn persistent_named_value(
        &self,
        function: &str,
        name: &str,
    ) -> Result<Option<Value>, crate::RuntimeError> {
        self.with_state(|state| state.persistent_named_value(function, name))
    }

    pub fn persistent_slot_value(
        &self,
        function: &str,
        slot: usize,
    ) -> Result<Option<Value>, crate::RuntimeError> {
        self.with_state(|state| state.persistent_slot_value(function, slot))
    }

    pub fn store_persistent_slot(
        &self,
        function: &str,
        slot: usize,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(|state| state.store_persistent_slot(function, slot, value))
    }

    pub fn persistent_revision(&self) -> Result<PersistentRevision, crate::RuntimeError> {
        self.with_state(|state| state.persistent_revision.clone())
    }

    pub fn store_persistent_named(
        &self,
        function: &str,
        name: &str,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(|state| state.store_persistent_named(function, name, value))
    }

    pub fn store_persistent_named_with_context(
        &self,
        context: &crate::context::RuntimeContext,
        function: &str,
        name: &str,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        self.validate_context(context)?;
        self.store_persistent_named(function, name, value)
    }

    pub fn store_persistent_named_and_slot(
        &self,
        function: &str,
        name: &str,
        slot: usize,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(|state| {
            state
                .persistent_slots
                .insert((function.to_string(), slot), value.clone());
            state
                .persistent_names
                .insert((function.to_string(), name.to_string()), value);
            state.persistent_revision = PersistentRevision::default();
        })
    }

    pub fn persistent_invalidation_revision(
        &self,
        function: &str,
    ) -> Result<PersistentInvalidationRevision, crate::RuntimeError> {
        self.with_state_mut(|state| state.persistent_invalidation_revision(function))
    }

    pub fn clear_persistent_function(&self, function: &str) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(|state| state.clear_persistent_function(function))
    }

    pub fn clear_all_persistents(&self) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(SessionVariableState::clear_all_persistents)
    }
}

/// Opaque version of the selected session's persistent store. A new Rc on
/// mutation/reset avoids ABA even when an old native frame survives a reset.
#[derive(Clone, Debug)]
pub struct PersistentRevision(Rc<()>);

impl Default for PersistentRevision {
    fn default() -> Self {
        Self(Rc::new(()))
    }
}

impl PartialEq for PersistentRevision {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for PersistentRevision {}

/// Tracks successful function invalidation separately from ordinary persistent
/// writes, including clears of a function whose first declaration has not yet
/// created a canonical value.
#[derive(Clone, Debug)]
pub struct PersistentInvalidationRevision(Rc<()>);

impl PartialEq for PersistentInvalidationRevision {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for PersistentInvalidationRevision {}

/// Native-to-VM continuation of one declared persistent alias. The retained
/// invalidation lease is the frame's authority, not a token reconstructed at
/// deoptimization time after a possible function clear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistentBindingContinuation {
    pub name: String,
    pub function: String,
    pub owner: SessionVariableAuthority,
    pub invalidation: PersistentInvalidationRevision,
    pub invalidated: bool,
}

impl PersistentBindingContinuation {
    pub fn validate_selected(
        &self,
        authority: &SessionVariableAuthority,
        function: &str,
    ) -> Result<(), crate::RuntimeError> {
        if &self.owner != authority || self.function != function {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Workspace:PersistentResumeOwner",
                "persistent continuation belongs to a different session or function",
            ));
        }
        self.owner.with_state(|_| ())
    }
}

impl SessionVariableState {
    pub(super) fn persistent_invalidation_revision(
        &mut self,
        function: &str,
    ) -> PersistentInvalidationRevision {
        if let Some(token) = self
            .persistent_invalidation_leases
            .get(function)
            .and_then(Weak::upgrade)
        {
            return PersistentInvalidationRevision(token);
        }
        // Dead names are not authority. Prune only when a new function lease
        // is needed, keeping ordinary aliased reads O(1).
        self.persistent_invalidation_leases
            .retain(|_, token| token.strong_count() > 0);
        let token = Rc::new(());
        self.persistent_invalidation_leases
            .insert(function.to_string(), Rc::downgrade(&token));
        PersistentInvalidationRevision(token)
    }

    pub(super) fn clear_persistent_function(&mut self, function: &str) {
        self.persistent_slots
            .retain(|(owner, _), _| owner != function);
        self.persistent_names
            .retain(|(owner, _), _| owner != function);
        self.persistent_invalidation_leases.remove(function);
        self.persistent_revision = PersistentRevision::default();
    }

    pub(super) fn clear_all_persistents(&mut self) {
        self.persistent_slots.clear();
        self.persistent_names.clear();
        self.persistent_invalidation_leases.clear();
        self.persistent_revision = PersistentRevision::default();
    }

    pub(super) fn persistent_slot_value(&self, function: &str, slot: usize) -> Option<Value> {
        self.persistent_slots
            .get(&(function.to_string(), slot))
            .cloned()
    }
    pub(super) fn persistent_named_value(&self, function: &str, name: &str) -> Option<Value> {
        self.persistent_names
            .get(&(function.to_string(), name.to_string()))
            .cloned()
    }

    pub(super) fn update_persistent_slot(&mut self, function: &str, index: usize, value: &Value) {
        let key = (function.to_string(), index);
        if let Some(existing) = self.persistent_slots.get_mut(&key) {
            *existing = value.clone();
            self.persistent_revision = PersistentRevision::default();
        }
    }

    pub(super) fn store_persistent_slot(&mut self, function: &str, index: usize, value: Value) {
        self.persistent_slots
            .insert((function.to_string(), index), value);
        self.persistent_revision = PersistentRevision::default();
    }

    pub(super) fn store_persistent_named(&mut self, function: &str, name: &str, value: Value) {
        self.persistent_names
            .insert((function.to_string(), name.to_string()), value);
        self.persistent_revision = PersistentRevision::default();
    }
}

pub fn persistent_slot_value(function: &str, index: usize) -> Option<Value> {
    with_state(|state| state.persistent_slot_value(function, index))
}

pub fn persistent_named_value(function: &str, name: &str) -> Option<Value> {
    with_state(|state| state.persistent_named_value(function, name))
}

pub fn persistent_revision() -> PersistentRevision {
    with_state(|state| state.persistent_revision.clone())
}

pub fn update_persistent_slot(function: &str, index: usize, value: &Value) {
    with_state_mut(|state| state.update_persistent_slot(function, index, value));
}

pub fn store_persistent_slot(function: &str, index: usize, value: Value) {
    with_state_mut(|state| state.store_persistent_slot(function, index, value));
}

pub fn store_persistent_named(function: &str, name: &str, value: Value) {
    with_state_mut(|state| state.store_persistent_named(function, name, value));
}
