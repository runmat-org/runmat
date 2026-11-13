//! Frame-owned persistent aliases. Stack operands remain evaluated snapshots;
//! only an aliasing variable slot is refreshed from canonical session storage.

use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::{workspace::session::SessionVariableAuthority, RuntimeError};
use runmat_value::Value;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

pub type PersistentAliases = HashMap<usize, PersistentBinding>;

#[derive(Clone, Debug)]
pub struct PersistentBinding {
    alias: PersistentAlias,
    owner: SessionVariableAuthority,
    function: String,
    revision: RefCell<runmat_runtime::workspace::session::PersistentRevision>,
    invalidation: RefCell<runmat_runtime::workspace::session::PersistentInvalidationRevision>,
    invalidated: Cell<bool>,
}

impl PartialEq for PersistentBinding {
    fn eq(&self, other: &Self) -> bool {
        self.alias == other.alias
            && self.owner == other.owner
            && self.function == other.function
            && *self.revision.borrow() == *other.revision.borrow()
            && *self.invalidation.borrow() == *other.invalidation.borrow()
            && self.invalidated.get() == other.invalidated.get()
    }
}

impl Eq for PersistentBinding {}

impl PersistentBinding {
    pub(crate) fn matches_inherited_owner(&self, function: &str, name: &str) -> bool {
        self.function == function
            && matches!(&self.alias,
                PersistentAlias::InheritedNamed(alias) if alias == name)
    }

    pub fn from_native_continuation(
        continuation: runmat_runtime::workspace::session::PersistentBindingContinuation,
        authority: &SessionVariableAuthority,
        function: &str,
    ) -> Result<Self, RuntimeError> {
        Self::from_native_continuation_with_role(continuation, authority, function, false)
    }

    pub fn from_inherited_native_continuation(
        continuation: runmat_runtime::workspace::session::PersistentBindingContinuation,
        authority: &SessionVariableAuthority,
        function: &str,
    ) -> Result<Self, RuntimeError> {
        Self::from_native_continuation_with_role(continuation, authority, function, true)
    }

    fn from_native_continuation_with_role(
        continuation: runmat_runtime::workspace::session::PersistentBindingContinuation,
        authority: &SessionVariableAuthority,
        function: &str,
        inherited: bool,
    ) -> Result<Self, RuntimeError> {
        continuation.validate_selected(authority, function)?;
        Ok(Self {
            alias: if inherited {
                PersistentAlias::InheritedNamed(continuation.name)
            } else {
                PersistentAlias::Named(continuation.name)
            },
            owner: continuation.owner,
            function: continuation.function,
            // Force the first resumed alias read to consult the selected
            // authority. Materialization may precede a function clear; using
            // the current revision here would hide that invalidation.
            revision: RefCell::new(Default::default()),
            invalidation: RefCell::new(continuation.invalidation),
            invalidated: Cell::new(continuation.invalidated),
        })
    }

    pub fn validate_selected(
        &self,
        authority: &SessionVariableAuthority,
        function: &str,
    ) -> Result<(), RuntimeError> {
        if &self.owner != authority || self.function != function {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Workspace:PersistentResumeOwner",
                "persistent frame binding belongs to a different session or function",
            ));
        }
        self.owner.persistent_revision().map(|_| ())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistentAlias {
    Named(String),
    /// A child frame delegates to its ancestor's named storage. Its local slot
    /// is not the declaring function's legacy slot and must never be used as
    /// a fallback or a second publication key.
    InheritedNamed(String),
    LegacySlot,
}

fn canonical(
    authority: &SessionVariableAuthority,
    function: &str,
    slot: usize,
    alias: &PersistentAlias,
) -> Result<Option<Value>, RuntimeError> {
    match alias {
        PersistentAlias::Named(name) => match authority.persistent_named_value(function, name)? {
            Some(value) => Ok(Some(value)),
            None => authority.persistent_slot_value(function, slot),
        },
        PersistentAlias::InheritedNamed(name) => authority.persistent_named_value(function, name),
        PersistentAlias::LegacySlot => authority.persistent_slot_value(function, slot),
    }
}

fn publish(
    authority: &SessionVariableAuthority,
    function: &str,
    slot: usize,
    alias: &PersistentAlias,
    value: Value,
) -> Result<(), RuntimeError> {
    match alias {
        PersistentAlias::Named(name) => {
            authority.store_persistent_named_and_slot(function, name, slot, value)
        }
        PersistentAlias::InheritedNamed(name) => {
            authority.store_persistent_named(function, name, value)
        }
        PersistentAlias::LegacySlot => authority.store_persistent_slot(function, slot, value),
    }
}

pub(crate) fn declare(
    aliases: &mut PersistentAliases,
    vars: &mut InterpreterVars,
    authority: &SessionVariableAuthority,
    function: &str,
    slot: usize,
    alias: PersistentAlias,
) -> Result<(), RuntimeError> {
    let initial = canonical(authority, function, slot, &alias)?;
    set_slot(vars, slot, initial.unwrap_or_else(empty_workspace_value));
    aliases.insert(
        slot,
        PersistentBinding {
            alias,
            owner: authority.clone(),
            function: function.to_string(),
            revision: RefCell::new(authority.persistent_revision()?),
            invalidation: RefCell::new(authority.persistent_invalidation_revision(function)?),
            invalidated: Cell::new(false),
        },
    );
    Ok(())
}

pub(crate) fn refresh(
    aliases: &PersistentAliases,
    vars: &mut InterpreterVars,
    authority: &SessionVariableAuthority,
    _function: &str,
    slot: usize,
) -> Result<(), RuntimeError> {
    if let Some(binding) = aliases.get(&slot) {
        let revision = authority.persistent_revision()?;
        if *binding.revision.borrow() == revision {
            return Ok(());
        }
        let function = binding.function.as_str();
        let invalidation = authority.persistent_invalidation_revision(function)?;
        let was_cleared = *binding.invalidation.borrow() != invalidation;
        match canonical(authority, function, slot, &binding.alias)? {
            Some(value) => {
                set_slot(vars, slot, value);
                binding.invalidated.set(false);
            }
            None if was_cleared => {
                set_slot(vars, slot, empty_workspace_value());
                binding.invalidated.set(true);
            }
            None => {}
        }
        *binding.invalidation.borrow_mut() = invalidation;
        *binding.revision.borrow_mut() = revision;
    }
    Ok(())
}

pub(crate) fn publish_store(
    aliases: &PersistentAliases,
    vars: &InterpreterVars,
    authority: &SessionVariableAuthority,
    _function: &str,
    slot: usize,
) -> Result<(), RuntimeError> {
    if let (Some(binding), Some(value)) = (aliases.get(&slot), vars.get(slot)) {
        let function = binding.function.as_str();
        publish(authority, function, slot, &binding.alias, value.clone())?;
        binding.invalidated.set(false);
        *binding.revision.borrow_mut() = authority.persistent_revision()?;
        *binding.invalidation.borrow_mut() =
            authority.persistent_invalidation_revision(function)?;
    }
    Ok(())
}

/// Success-only completion neither replays old frame values nor leaves a
/// newly declared, never-written persistent absent from canonical storage.
pub(crate) fn finish_success(
    aliases: &PersistentAliases,
    vars: &mut InterpreterVars,
    authority: &SessionVariableAuthority,
    _function: &str,
) -> Result<(), RuntimeError> {
    let mut slots = aliases.keys().copied().collect::<Vec<_>>();
    slots.sort_unstable();
    for slot in slots {
        let binding = &aliases[&slot];
        let function = binding.function.as_str();
        let alias = &binding.alias;
        let value = canonical(authority, function, slot, alias)?;
        let invalidation = authority.persistent_invalidation_revision(function)?;
        let was_cleared =
            binding.invalidated.get() || *binding.invalidation.borrow() != invalidation;
        match value {
            Some(value) => {
                binding.invalidated.set(false);
                if let PersistentAlias::Named(name) | PersistentAlias::InheritedNamed(name) = alias
                {
                    if authority.persistent_named_value(function, name)?.is_none() {
                        publish(authority, function, slot, alias, value.clone())?;
                    }
                }
                set_slot(vars, slot, value);
            }
            None => {
                if was_cleared {
                    set_slot(vars, slot, empty_workspace_value());
                    binding.invalidated.set(true);
                } else if let Some(value) = vars.get(slot) {
                    publish(authority, function, slot, alias, value.clone())?;
                }
            }
        }
        *binding.revision.borrow_mut() = authority.persistent_revision()?;
        *binding.invalidation.borrow_mut() =
            authority.persistent_invalidation_revision(function)?;
    }
    Ok(())
}

fn empty_workspace_value() -> Value {
    Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0]).expect("canonical empty workspace shape"),
    )
}

fn set_slot(vars: &mut InterpreterVars, slot: usize, value: Value) {
    if slot >= vars.len() {
        vars.resize_owned(slot + 1, Value::Num(0.0));
    }
    let _retired = vars
        .replace_owned(slot, value)
        .expect("slot was resized before replacement");
    crate::runtime::workspace::refresh_workspace_slot_from_vars(slot, vars);
}

#[cfg(test)]
#[path = "persistent/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "persistent/initialization_tests.rs"]
mod initialization_tests;

#[cfg(test)]
#[path = "persistent/inherited_tests.rs"]
mod inherited_tests;
