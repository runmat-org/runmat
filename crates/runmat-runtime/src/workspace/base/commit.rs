use super::delta::{WorkspaceChangeData, WorkspaceChangeStorage};
use super::owner::{BaseWorkspaceCommit, BaseWorkspaceState};
use super::storage::{ManagedWorkspaceStorage, WorkspaceNamespaceAction};
use super::values::RootedWorkspaceEntry;
use super::{BaseWorkspaceError, WorkspaceDelta, WorkspaceRevision};
use crate::workspace::retained::WorkspaceName;
use crate::workspace::GlobalBindingTarget;
use std::collections::BTreeMap;

/// Final state of one name touched by an explicitly unbounded direct delta.
/// The authentic input delta remains live until after the Base borrow retires,
/// so intermediate Foreign values never release during this preparation.
struct DirectNameState {
    local: Option<RootedWorkspaceEntry>,
    global: Option<GlobalBindingTarget>,
    changed: bool,
}

/// Off-state direct-owner transaction. The original delta remains alive while
/// its independently owned canonical effect is normalized; ignored or
/// intermediate Foreign values therefore cannot retire under a Base borrow.
pub(super) struct PreparedDirectCommit {
    base_revision: WorkspaceRevision,
    expected_revision: WorkspaceRevision,
    next_revision: WorkspaceRevision,
    storage: ManagedWorkspaceStorage,
    effect: WorkspaceDelta,
}

impl PreparedDirectCommit {
    pub(super) fn expected_revision(&self) -> WorkspaceRevision {
        self.expected_revision
    }

    pub(super) fn canonicalize_effect(&mut self) {
        if !self.effect.is_prepared_assignments_only() {
            self.effect.canonicalize_effect(|name| {
                (
                    self.storage.entry(name.as_str()),
                    self.storage.global_target(name.as_str()).cloned(),
                    WorkspaceChangeStorage::ManagedOverlay,
                )
            });
        }
    }
}

impl BaseWorkspaceState {
    pub(super) fn prepare_projection_commit(
        &self,
        base_revision: WorkspaceRevision,
        original: &WorkspaceDelta,
        effect: WorkspaceDelta,
        storage: ManagedWorkspaceStorage,
    ) -> Result<PreparedDirectCommit, BaseWorkspaceError> {
        let next_revision = self.preflight_commit(base_revision, original)?;
        assert!(self.entries.is_empty() && self.global_bindings.is_empty());
        Ok(PreparedDirectCommit {
            base_revision,
            expected_revision: self.revision,
            next_revision,
            storage,
            effect,
        })
    }

    pub(super) fn prepare_direct_commit(
        &self,
        base_revision: WorkspaceRevision,
        original: &WorkspaceDelta,
    ) -> Result<PreparedDirectCommit, BaseWorkspaceError> {
        let mut effect = original.clone_unbounded();
        let expected_revision = self.revision;
        let next = self.preflight_commit(base_revision, original)?;
        let mut storage = self
            .managed_storage
            .prepare_unbounded_capture(&self.entries, &self.global_bindings)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        if !effect.is_empty() {
            let mut clear = false;
            let mut names = BTreeMap::<String, DirectNameState>::new();
            for change in effect.changes_mut() {
                let name = match change.data() {
                    WorkspaceChangeData::Clear => {
                        clear = true;
                        names.clear();
                        change.mark_staged_in_managed_overlay();
                        continue;
                    }
                    WorkspaceChangeData::Assign { name, .. }
                    | WorkspaceChangeData::Remove { name }
                    | WorkspaceChangeData::BindGlobal { name, .. }
                    | WorkspaceChangeData::UnbindGlobal { name } => name.as_str(),
                };
                let state = names
                    .entry(name.to_owned())
                    .or_insert_with(|| DirectNameState {
                        local: (!clear).then(|| storage.entry(name)).flatten(),
                        global: (!clear)
                            .then(|| storage.global_target(name).cloned())
                            .flatten(),
                        changed: false,
                    });
                let staged = match change.data() {
                    WorkspaceChangeData::Assign { entry, .. } if state.global.is_none() => {
                        state.local = Some(entry.clone());
                        state.changed = true;
                        true
                    }
                    WorkspaceChangeData::Remove { .. } if state.global.is_none() => {
                        state.local = None;
                        state.changed = true;
                        true
                    }
                    WorkspaceChangeData::BindGlobal { target, .. } => {
                        state.local = None;
                        state.global = Some(target.clone());
                        state.changed = true;
                        true
                    }
                    WorkspaceChangeData::UnbindGlobal { .. } => {
                        state.local = None;
                        state.global = None;
                        state.changed = true;
                        true
                    }
                    WorkspaceChangeData::Assign { .. } | WorkspaceChangeData::Remove { .. } => {
                        false
                    }
                    WorkspaceChangeData::Clear => unreachable!("handled clear above"),
                };
                if staged {
                    change.mark_staged_in_managed_overlay();
                }
            }
            if clear {
                storage.stage_unbounded_clear();
            }
            for (name, state) in names {
                if !state.changed {
                    continue;
                }
                let name = WorkspaceName::unbounded(&name);
                match (state.local, state.global) {
                    (_, Some(target)) => storage.stage_unbounded_namespace(
                        name,
                        WorkspaceNamespaceAction::BindGlobal,
                        Some(target),
                    ),
                    (Some(entry), None) => storage.stage_unbounded_owned_assignment(name, entry),
                    (None, None) => storage.stage_unbounded_namespace(
                        name,
                        WorkspaceNamespaceAction::UnbindGlobal,
                        None,
                    ),
                }
            }
            storage
                .commit_unbounded_pending()
                .map_err(BaseWorkspaceError::HostAllocation)?;
        }
        Ok(PreparedDirectCommit {
            base_revision,
            expected_revision,
            next_revision: next,
            storage,
            effect,
        })
    }

    /// Caller validated the exact revision after off-state normalization.
    /// Only owner swaps and name-only journal publication occur under borrow.
    pub(super) fn install_direct_commit(
        &mut self,
        mut prepared: PreparedDirectCommit,
    ) -> (BaseWorkspaceCommit, ManagedWorkspaceStorage) {
        assert_eq!(self.revision, prepared.expected_revision);
        assert!(self.entries.is_empty() && self.global_bindings.is_empty());
        let retired_storage = std::mem::replace(&mut self.managed_storage, prepared.storage);
        let previous_revision = self.revision;
        self.revision = prepared.next_revision;
        if !prepared.effect.is_empty() {
            self.journal.record(
                previous_revision,
                prepared.next_revision,
                &mut prepared.effect,
            );
        }
        let commit = BaseWorkspaceCommit::new(
            prepared.base_revision,
            previous_revision,
            prepared.next_revision,
            prepared.effect,
        );
        (commit, retired_storage)
    }

    pub(super) fn preflight_commit(
        &self,
        base_revision: WorkspaceRevision,
        delta: &WorkspaceDelta,
    ) -> Result<WorkspaceRevision, BaseWorkspaceError> {
        self.ensure_open()?;
        if base_revision > self.revision {
            return Err(BaseWorkspaceError::FutureRevision {
                requested: base_revision.get(),
                current: self.revision.get(),
            });
        }
        if delta.is_empty() {
            return Ok(self.revision);
        }
        self.revision
            .next()
            .ok_or(BaseWorkspaceError::RevisionExhausted)
    }
}
