use std::collections::{BTreeSet, HashSet, VecDeque};

use super::delta::WorkspaceChangeData;
use super::values::RootedWorkspaceEntries;
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
}

#[derive(Debug)]
struct WorkspaceHistoryRecord {
    revision: WorkspaceRevision,
    impacts: Vec<WorkspaceImpact>,
}

#[derive(Debug)]
enum WorkspaceImpact {
    Name(String),
    Clear,
}

impl WorkspaceJournal {
    pub(super) fn new(change_limit: usize) -> Self {
        Self {
            floor: WorkspaceRevision::INITIAL,
            records: VecDeque::new(),
            change_count: 0,
            change_limit,
        }
    }

    pub(super) fn clear_at(&mut self, revision: WorkspaceRevision) {
        self.floor = revision;
        self.records.clear();
        self.change_count = 0;
    }

    pub(super) fn record(&mut self, revision: WorkspaceRevision, delta: &WorkspaceDelta) {
        let impacts = delta
            .changes()
            .iter()
            .map(|change| match change.data() {
                WorkspaceChangeData::Assign { name, .. }
                | WorkspaceChangeData::Remove { name }
                | WorkspaceChangeData::BindGlobal { name }
                | WorkspaceChangeData::UnbindGlobal { name } => WorkspaceImpact::Name(name.clone()),
                WorkspaceChangeData::Clear => WorkspaceImpact::Clear,
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
        global_bindings: &HashSet<String>,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        if revision > current {
            return Err(BaseWorkspaceError::FutureRevision {
                requested: revision.get(),
                current: current.get(),
            });
        }
        Ok(self.reconcile_known(revision, current, entries, global_bindings))
    }

    pub(super) fn reconcile_known(
        &self,
        revision: WorkspaceRevision,
        current: WorkspaceRevision,
        entries: &RootedWorkspaceEntries,
        global_bindings: &HashSet<String>,
    ) -> BaseWorkspaceReconciliation {
        if revision == current {
            return BaseWorkspaceReconciliation::UpToDate { revision };
        }
        if revision < self.floor {
            return BaseWorkspaceReconciliation::Reset(BaseWorkspaceSnapshot::new(
                current,
                entries.clone(),
                global_bindings.clone(),
            ));
        }

        let mut clear = false;
        let mut names = BTreeSet::new();
        for record in self
            .records
            .iter()
            .filter(|entry| entry.revision > revision)
        {
            for impact in &record.impacts {
                match impact {
                    WorkspaceImpact::Name(name) => {
                        names.insert(name.clone());
                    }
                    WorkspaceImpact::Clear => {
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
            if global_bindings.contains(&name) {
                changes.bind_global(name);
                continue;
            }
            changes.unbind_global(name.clone());
            match entries.get(&name) {
                Some(entry) => changes.push_assign_entry(name, entry.clone()),
                None if !clear => changes.remove(name),
                None => {}
            }
        }
        BaseWorkspaceReconciliation::Delta {
            from: revision,
            to: current,
            changes,
        }
    }
}
