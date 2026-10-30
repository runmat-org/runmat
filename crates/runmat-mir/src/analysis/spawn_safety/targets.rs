//! Provenance of a future operand, not permission to schedule it.
use super::{FunctionId, MirBody, MirLocalKind};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct FutureTargets {
    known: BTreeSet<FunctionId>,
    runtime_input: bool,
}

impl FutureTargets {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn runtime_checked() -> Self {
        Self {
            known: BTreeSet::new(),
            runtime_input: true,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.known.is_empty() && !self.runtime_input
    }
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.known.len()
    }
    pub fn clear(&mut self) {
        self.known.clear();
        self.runtime_input = false;
    }
    pub fn join(&mut self, other: &Self) {
        if self.is_empty() || other.is_empty() {
            self.clear();
        } else {
            self.known.extend(other.known.iter().copied());
            self.runtime_input |= other.runtime_input;
        }
    }
    pub fn known(&self) -> impl Iterator<Item = FunctionId> + '_ {
        self.known.iter().copied()
    }
    pub fn requires_runtime_owner(&self) -> bool {
        self.runtime_input
    }
}

impl<const N: usize> From<[FunctionId; N]> for FutureTargets {
    fn from(functions: [FunctionId; N]) -> Self {
        Self {
            known: functions.into_iter().collect(),
            runtime_input: false,
        }
    }
}

pub(super) fn entry_targets(body: &MirBody) -> Vec<FutureTargets> {
    body.locals
        .iter()
        .map(|local| FutureTargets {
            known: BTreeSet::new(),
            // This describes where a value arrives. The actual service-owned
            // Future and its captured spawn contract are checked at execution.
            runtime_input: matches!(
                local.kind,
                MirLocalKind::IncomingWorkspace | MirLocalKind::Parameter
            ),
        })
        .collect()
}
