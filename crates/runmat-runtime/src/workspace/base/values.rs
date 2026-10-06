use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_value::Value;

use crate::execution::RootedValueHandles;

use super::BaseWorkspaceError;

/// One immutable workspace binding. Clones share both the language value and
/// its managed-handle roots; mutation replaces the complete binding.
#[derive(Clone, Debug)]
pub(super) struct RootedWorkspaceEntry {
    value: Rc<Value>,
    _roots: RootedValueHandles,
}

impl RootedWorkspaceEntry {
    pub(super) fn new(value: Value) -> Result<Self, BaseWorkspaceError> {
        runmat_value::validate_no_transient_sequence(&value)?;
        let value = Rc::new(value);
        let roots = RootedValueHandles::for_values(std::iter::once(value.as_ref()))?;
        Ok(Self {
            value,
            _roots: roots,
        })
    }

    pub(super) fn value(&self) -> &Value {
        self.value.as_ref()
    }

    pub(super) fn clone_value(&self) -> Value {
        self.value.as_ref().clone()
    }

    #[cfg(test)]
    pub(super) fn shares_value_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.value, &other.value)
    }
}

pub(super) type RootedWorkspaceEntries = BTreeMap<String, RootedWorkspaceEntry>;
