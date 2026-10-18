mod impact;
pub(super) use impact::{PreparedWorkspaceImpacts, WorkspaceImpact};

use crate::workspace::GlobalBindingTarget;
use std::collections::{BTreeSet, HashMap, VecDeque};

use super::delta::{WorkspaceChange, WorkspaceChangeData, WorkspaceChangeStorage};
use super::storage::ManagedWorkspaceStorage;
use super::values::{RootedWorkspaceEntries, WorkspaceEntryName};
use super::{
    BaseWorkspaceError, BaseWorkspaceReconciliation, BaseWorkspaceSnapshot, WorkspaceDelta,
    WorkspaceRevision,
};

pub(super) const DEFAULT_HISTORY_CHANGE_LIMIT: usize = 4_096;

#[derive(Debug)]
pub(super) struct WorkspaceJournal {
    floor: WorkspaceRevision,
    records: VecDeque<WorkspaceHistoryRecord>,
    change_count: usize,
    change_limit: usize,
    managed: PreparedWorkspaceImpacts,
}

#[derive(Debug)]
struct WorkspaceHistoryRecord {
    revision: WorkspaceRevision,
    impacts: Vec<LegacyWorkspaceImpact>,
}

#[derive(Debug)]
enum LegacyWorkspaceImpact {
    Name(WorkspaceEntryName),
    Clear,
}

impl WorkspaceJournal {
    pub(super) fn new(change_limit: usize) -> Self {
        Self {
            floor: WorkspaceRevision::INITIAL,
            records: VecDeque::new(),
            change_count: 0,
            change_limit,
            managed: PreparedWorkspaceImpacts::default(),
        }
    }

    pub(super) fn clear_at(&mut self, revision: WorkspaceRevision) {
        self.floor = revision;
        self.records.clear();
        self.change_count = 0;
        self.managed = PreparedWorkspaceImpacts::default();
    }

    pub(super) fn record(
        &mut self,
        previous_revision: WorkspaceRevision,
        revision: WorkspaceRevision,
        delta: &mut WorkspaceDelta,
    ) {
        let reset_mixed_history = delta.requires_history_reset();
        let mut managed = delta.take_prepared_impacts();
        if reset_mixed_history {
            // A context-bearing mutation may not fall through to the legacy
            // allocation path when an earlier mutation in the same delta was
            // not covered by the pre-admitted impact chain. Reset is the
            // conservative complete reconciliation result.
            self.clear_at(revision);
            return;
        }
        if !managed.is_empty() {
            if !self.records.is_empty() {
                self.records.clear();
                self.change_count = 0;
                self.floor = previous_revision;
            }
            managed.stamp(revision);
            managed.attach_older(std::mem::take(&mut self.managed));
            self.change_count = managed.len();
            if let Some(floor) = managed.truncate(self.change_limit) {
                self.floor = floor;
            }
            self.change_count = managed.len();
            self.managed = managed;
            return;
        }
        if !self.managed.is_empty() {
            self.managed = PreparedWorkspaceImpacts::default();
            self.change_count = 0;
            self.floor = previous_revision;
        }
        let impacts = delta
            .changes()
            .iter()
            .map(|change| match change.data() {
                WorkspaceChangeData::Assign { name, .. }
                | WorkspaceChangeData::Remove { name }
                | WorkspaceChangeData::BindGlobal { name, .. }
                | WorkspaceChangeData::UnbindGlobal { name } => {
                    LegacyWorkspaceImpact::Name(name.clone())
                }
                WorkspaceChangeData::Clear => LegacyWorkspaceImpact::Clear,
            })
            .collect::<Vec<_>>();
        let impact_count = impacts.len();
        if impact_count > self.change_limit || self.change_limit == 0 {
            self.clear_at(revision);
            return;
        }
        self.change_count += impact_count;
        self.records
            .push_back(WorkspaceHistoryRecord { revision, impacts });
        while self.change_count > self.change_limit {
            if let Some(retired) = self.records.pop_front() {
                self.change_count -= retired.impacts.len();
                self.floor = retired.revision;
            }
        }
    }

    pub(super) fn reconcile(
        &self,
        revision: WorkspaceRevision,
        current: WorkspaceRevision,
        entries: &RootedWorkspaceEntries,
        global_bindings: &HashMap<String, GlobalBindingTarget>,
        managed_storage: &ManagedWorkspaceStorage,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        if revision > current {
            return Err(BaseWorkspaceError::FutureRevision {
                requested: revision.get(),
                current: current.get(),
            });
        }
        Ok(self.reconcile_known(revision, current, entries, global_bindings, managed_storage))
    }

    pub(super) fn reconcile_known(
        &self,
        revision: WorkspaceRevision,
        current: WorkspaceRevision,
        entries: &RootedWorkspaceEntries,
        global_bindings: &HashMap<String, GlobalBindingTarget>,
        managed_storage: &ManagedWorkspaceStorage,
    ) -> BaseWorkspaceReconciliation {
        if revision == current {
            return BaseWorkspaceReconciliation::UpToDate { revision };
        }
        if revision < self.floor {
            return BaseWorkspaceReconciliation::Reset(BaseWorkspaceSnapshot::new(
                current,
                entries.clone(),
                global_bindings.clone(),
                managed_storage.clone(),
            ));
        }

        let mut clear = false;
        let mut names = BTreeSet::new();
        for (record_revision, impact) in self.managed.iter_newest() {
            if record_revision <= revision {
                break;
            }
            match impact {
                WorkspaceImpact::Name(name) => {
                    names.insert(name.clone());
                }
                WorkspaceImpact::Clear => {
                    clear = true;
                    break;
                }
            }
        }
        for record in self
            .records
            .iter()
            .filter(|entry| entry.revision > revision)
        {
            for impact in &record.impacts {
                match impact {
                    LegacyWorkspaceImpact::Name(name) => {
                        names.insert(name.clone());
                    }
                    LegacyWorkspaceImpact::Clear => {
                        clear = true;
                        names.clear();
                    }
                }
            }
        }

        let mut changes = WorkspaceDelta::new();
        if clear {
            changes.clear();
        }
        for name in names {
            let target = match managed_storage.global(name.as_str()) {
                Some(true) => managed_storage.global_target(name.as_str()),
                Some(false) => None,
                None => global_bindings.get(name.as_str()),
            };
            if let Some(target) = target {
                let storage = if managed_storage.global(name.as_str()) == Some(true) {
                    WorkspaceChangeStorage::ManagedOverlay
                } else {
                    WorkspaceChangeStorage::Legacy
                };
                changes.push_reconciled_change(
                    WorkspaceChange::bind_global_prepared(name, target.clone())
                        .with_storage(storage),
                );
                continue;
            }
            changes.push_reconciled_change(WorkspaceChange::unbind_global_prepared(name.clone()));
            if let Some(entry) = managed_storage.entry(name.as_str()) {
                changes.push_reconciled_change(
                    WorkspaceChange::assign_entry(name, entry)
                        .with_storage(WorkspaceChangeStorage::ManagedOverlay),
                );
            } else if let Some(entry) = entries.get(&name).cloned() {
                changes.push_reconciled_change(WorkspaceChange::assign_entry(name, entry));
            } else if !clear {
                changes.push_reconciled_change(WorkspaceChange::remove_prepared(name));
            }
        }
        BaseWorkspaceReconciliation::Delta {
            from: revision,
            to: current,
            changes,
        }
    }
}
