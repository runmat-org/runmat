//! Binding-origin facts accompanying successful canonical workspace commits.
//!
//! Values remain owned by the projection and base workspace. These events
//! carry only the origin and exact executor slot of the final mutation.

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use crate::RuntimeError;

use super::super::{BaseWorkspaceCommit, WorkspaceChange, WorkspaceChangeKind, WorkspaceDelta};

#[derive(Debug, Default)]
pub(super) struct PublishedWorkspaceEffect {
    cleared: bool,
    changes: BTreeMap<String, WorkspaceChange>,
}

impl PublishedWorkspaceEffect {
    pub(super) fn observe(&mut self, delta: &WorkspaceDelta) -> Vec<WorkspaceChange> {
        let mut retired = Vec::new();
        for change in delta.changes() {
            match change.kind() {
                WorkspaceChangeKind::Clear => {
                    self.cleared = true;
                    retired.extend(std::mem::take(&mut self.changes).into_values());
                }
                WorkspaceChangeKind::Assign | WorkspaceChangeKind::Remove => {
                    if let Some(name) = change.name() {
                        if let Some(previous) =
                            self.changes.insert(name.to_string(), change.clone())
                        {
                            retired.push(previous);
                        }
                    }
                }
            }
        }
        retired
    }

    pub(super) fn take(&mut self) -> WorkspaceDelta {
        let mut changes = Vec::with_capacity(self.changes.len() + usize::from(self.cleared));
        if self.cleared {
            changes.push(WorkspaceChange::clear());
        }
        changes.extend(std::mem::take(&mut self.changes).into_values());
        self.cleared = false;
        WorkspaceDelta::from_changes(changes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspacePublicationOrigin {
    ExecutorSlot { name: String, slot: usize },
    DynamicEffect { name: String },
    Retired { name: String },
    Cleared,
}

type WorkspacePublicationCallback =
    dyn Fn(&[WorkspacePublicationOrigin]) -> Result<(), RuntimeError>;

#[derive(Clone)]
pub struct WorkspacePublicationObserver(Rc<WorkspacePublicationCallback>);

impl fmt::Debug for WorkspacePublicationObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("WorkspacePublicationObserver")
    }
}

impl WorkspacePublicationObserver {
    pub fn new(
        observer: impl Fn(&[WorkspacePublicationOrigin]) -> Result<(), RuntimeError> + 'static,
    ) -> Self {
        Self(Rc::new(observer))
    }

    pub(super) fn notify(
        &self,
        changes: &[WorkspacePublicationOrigin],
    ) -> Result<(), RuntimeError> {
        (self.0)(changes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingOrigin {
    ExecutorSlot(usize),
    DynamicEffect,
}

#[derive(Debug, Default)]
pub(super) struct PendingPublicationOrigins {
    assignments: BTreeMap<String, PendingOrigin>,
}

impl PendingPublicationOrigins {
    pub(super) fn executor_slot(&mut self, name: &str, slot: usize) {
        self.assignments
            .insert(name.to_string(), PendingOrigin::ExecutorSlot(slot));
    }

    pub(super) fn dynamic_effect(&mut self, name: &str) {
        self.assignments
            .insert(name.to_string(), PendingOrigin::DynamicEffect);
    }

    pub(super) fn retire(&mut self, name: &str) {
        self.assignments.remove(name);
    }

    pub(super) fn clear(&mut self) {
        self.assignments.clear();
    }

    pub(super) fn committed(
        &mut self,
        commit: &BaseWorkspaceCommit,
    ) -> Vec<WorkspacePublicationOrigin> {
        let mut changes = Vec::new();
        for change in commit.changes().changes() {
            match change.kind() {
                WorkspaceChangeKind::Clear => {
                    changes.push(WorkspacePublicationOrigin::Cleared);
                }
                WorkspaceChangeKind::Assign => {
                    let Some(name) = change.name() else { continue };
                    let origin = self.assignments.remove(name);
                    changes.push(match origin {
                        Some(PendingOrigin::ExecutorSlot(slot)) => {
                            WorkspacePublicationOrigin::ExecutorSlot {
                                name: name.into(),
                                slot,
                            }
                        }
                        Some(PendingOrigin::DynamicEffect) | None => {
                            WorkspacePublicationOrigin::DynamicEffect { name: name.into() }
                        }
                    });
                }
                WorkspaceChangeKind::Remove => {
                    let Some(name) = change.name() else { continue };
                    self.assignments.remove(name);
                    changes.push(WorkspacePublicationOrigin::Retired { name: name.into() });
                }
            }
        }
        self.assignments.clear();
        changes
    }
}

#[cfg(test)]
mod tests;
