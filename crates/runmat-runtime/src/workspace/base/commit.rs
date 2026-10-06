use super::delta::WorkspaceChangeData;
use super::owner::{BaseWorkspaceCommit, BaseWorkspaceState};
use super::values::RootedWorkspaceEntries;
use super::{BaseWorkspaceError, WorkspaceDelta, WorkspaceRevision};
use std::collections::BTreeSet;

impl BaseWorkspaceState {
    pub(super) fn commit(
        &mut self,
        base_revision: WorkspaceRevision,
        delta: WorkspaceDelta,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let next = self.preflight_commit(base_revision, &delta)?;
        Ok(self.commit_preflighted(base_revision, next, delta))
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

    pub(super) fn commit_preflighted(
        &mut self,
        base_revision: WorkspaceRevision,
        revision: WorkspaceRevision,
        delta: WorkspaceDelta,
    ) -> BaseWorkspaceCommit {
        let previous_revision = self.revision;
        if !delta.is_empty() {
            apply_delta(&mut self.entries, &delta);
            self.revision = revision;
            self.journal.record(revision, &delta);
        }
        let changes = canonical_effect(&delta, &self.entries);
        BaseWorkspaceCommit::new(base_revision, previous_revision, self.revision, changes)
    }
}

/// Collapse an ordered input into the canonical effect observed after commit.
/// Intermediate assigned values therefore stop retaining managed captures as
/// soon as publication completes, even when a receipt remains alive.
fn canonical_effect(delta: &WorkspaceDelta, entries: &RootedWorkspaceEntries) -> WorkspaceDelta {
    let mut clear = false;
    let mut names = BTreeSet::new();
    for change in delta.changes() {
        match change.data() {
            WorkspaceChangeData::Assign { name, .. } | WorkspaceChangeData::Remove { name } => {
                names.insert(name.clone());
            }
            WorkspaceChangeData::Clear => {
                clear = true;
                names.clear();
            }
        }
    }
    let mut effect = WorkspaceDelta::new();
    if clear {
        effect.clear();
    }
    for name in names {
        match entries.get(&name) {
            Some(entry) => effect.push_assign_entry(name, entry.clone()),
            None if !clear => effect.remove(name),
            None => {}
        }
    }
    effect
}

fn apply_delta(entries: &mut super::values::RootedWorkspaceEntries, delta: &WorkspaceDelta) {
    for change in delta.changes() {
        match change.data() {
            WorkspaceChangeData::Assign { name, entry } => {
                entries.insert(name.clone(), entry.clone());
            }
            WorkspaceChangeData::Remove { name } => {
                entries.remove(name);
            }
            WorkspaceChangeData::Clear => entries.clear(),
        }
    }
}
