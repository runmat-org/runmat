use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_execution::{CancellationReason, ExecutionScopeId};
use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancellationState {
    pub reason: CancellationReason,
    pub requested_at_millis: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancellationTree {
    parents: BTreeMap<ExecutionScopeId, Option<ExecutionScopeId>>,
    children: BTreeMap<ExecutionScopeId, BTreeSet<ExecutionScopeId>>,
    cancelled: BTreeMap<ExecutionScopeId, CancellationState>,
}

impl CancellationTree {
    pub fn register(
        &mut self,
        scope: ExecutionScopeId,
        parent: Option<ExecutionScopeId>,
    ) -> RunnerResult<()> {
        if self.parents.contains_key(&scope) {
            return Ok(());
        }
        if parent == Some(scope) {
            return Err(RunnerError::Invalid(
                "execution scope cannot parent itself".into(),
            ));
        }
        if let Some(parent) = parent {
            if !self.parents.contains_key(&parent) {
                return Err(RunnerError::Invalid(format!(
                    "parent execution scope {parent} is not registered"
                )));
            }
            self.children.entry(parent).or_default().insert(scope);
        }
        self.parents.insert(scope, parent);
        Ok(())
    }

    pub fn cancel(
        &mut self,
        scope: ExecutionScopeId,
        reason: CancellationReason,
        now_millis: u64,
    ) -> RunnerResult<Vec<ExecutionScopeId>> {
        if !self.parents.contains_key(&scope) {
            return Err(RunnerError::Invalid(format!(
                "execution scope {scope} is not registered"
            )));
        }
        let mut queue = VecDeque::from([scope]);
        let mut newly_cancelled = Vec::new();
        while let Some(current) = queue.pop_front() {
            if let std::collections::btree_map::Entry::Vacant(entry) = self.cancelled.entry(current)
            {
                entry.insert(CancellationState {
                    reason,
                    requested_at_millis: now_millis,
                });
                newly_cancelled.push(current);
            }
            queue.extend(self.children.get(&current).into_iter().flatten().copied());
        }
        Ok(newly_cancelled)
    }

    pub fn state(&self, scope: ExecutionScopeId) -> Option<CancellationState> {
        self.cancelled.get(&scope).copied()
    }

    pub fn contains(&self, scope: ExecutionScopeId) -> bool {
        self.parents.contains_key(&scope)
    }
}
