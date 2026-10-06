use std::collections::BTreeMap;

use runmat_value::Value;

use crate::execution::RootedValueHandles;

use super::BaseWorkspaceError;

#[derive(Clone, Debug)]
pub(super) struct RootedWorkspaceValues {
    values: BTreeMap<String, Value>,
    _roots: RootedValueHandles,
}

impl RootedWorkspaceValues {
    pub(super) fn empty() -> Result<Self, BaseWorkspaceError> {
        Self::new(BTreeMap::new())
    }

    pub(super) fn new(values: BTreeMap<String, Value>) -> Result<Self, BaseWorkspaceError> {
        for value in values.values() {
            runmat_value::validate_no_transient_sequence(value)?;
        }
        let roots = RootedValueHandles::for_values(values.values())?;
        Ok(Self {
            values,
            _roots: roots,
        })
    }

    pub(super) fn values(&self) -> &BTreeMap<String, Value> {
        &self.values
    }

    pub(super) fn remove(&mut self, name: &str) {
        self.values.remove(name);
    }

    pub(super) fn clear(&mut self) {
        self.values.clear();
    }

    pub(super) fn into_values(self) -> BTreeMap<String, Value> {
        self.values
    }
}
