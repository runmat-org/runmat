use super::*;

pub(in crate::workspace::base) struct PreparedProjectionCompletion {
    pub(super) entries: Option<RootedWorkspaceEntries>,
    pub(super) global_bindings: Option<HashMap<String, GlobalBindingTarget>>,
    pub(super) managed_storage: ManagedWorkspaceStorage,
    pub(super) pending: WorkspaceDelta,
    pub(super) captures_legacy: bool,
    pub(super) source_start: usize,
    pub(super) expected_revision: WorkspaceRevision,
    pub(super) expected_generation: u64,
}

impl PreparedProjectionCompletion {
    pub(in crate::workspace::base) fn prepare_effect_unbounded(&self) -> WorkspaceDelta {
        self.pending.clone_unbounded()
    }

    pub(in crate::workspace::base) fn prepare_effect(
        &self,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<WorkspaceDelta, crate::context::host_allocation::HostAllocationError> {
        self.pending.prepare_independent_effect(permit)
    }
}

pub(super) enum PreparedProjectionAction {
    Named { name: WorkspaceName },
    Clear,
}

/// One entirely prepared live generation and its authentic pending effect.
/// Installation swaps owners only; displaced roots retire after observers.
pub(in crate::workspace::base) struct PreparedProjectionMutation {
    pub(super) live: ManagedWorkspaceStorage,
    pub(super) pending: PreparedWorkspaceDeltaMutation,
    pub(super) action: PreparedProjectionAction,
    pub(super) captures_legacy: bool,
    pub(super) expected_revision: WorkspaceRevision,
    pub(super) expected_generation: u64,
}

enum PreparedPendingChanges<'a> {
    Mutation(PreparedWorkspaceDeltaView<'a>),
    Complete(&'a WorkspaceDelta),
}

/// Borrowed prospective state for an exact prepared mutation or completion.
/// No workspace value, name map, or pending Vec is copied to construct it.
pub(in crate::workspace::base) struct PreparedProjectionPublicationView<'a> {
    base: OrderedProjectionRef<'a>,
    pending: PreparedPendingChanges<'a>,
    managed_storage: Option<&'a ManagedWorkspaceStorage>,
    entries: Option<&'a RootedWorkspaceEntries>,
    globals: Option<&'a HashMap<String, GlobalBindingTarget>>,
    captures_legacy: bool,
    source_start: usize,
    expected_revision: WorkspaceRevision,
    expected_generation: u64,
}

impl PreparedProjectionPublicationView<'_> {
    pub(in crate::workspace::base) fn revision(&self) -> WorkspaceRevision {
        self.expected_revision
    }

    pub(in crate::workspace::base) fn before_generation(&self) -> u64 {
        self.expected_generation
    }

    pub(in crate::workspace::base) fn mutation_change(&self) -> Option<&WorkspaceChange> {
        match &self.pending {
            PreparedPendingChanges::Mutation(view) => Some(view.prepared_change()),
            PreparedPendingChanges::Complete(_) => None,
        }
    }

    pub(in crate::workspace::base) fn matches_base(&self) -> bool {
        self.base.revision() == self.expected_revision
            && self.base.pending().generation() == self.expected_generation
    }

    pub(in crate::workspace::base) fn changes(
        &self,
    ) -> impl ExactSizeIterator<Item = &WorkspaceChange> + '_ {
        (0..self.len()).map(|index| self.change(index).expect("prepared change index"))
    }

    pub(in crate::workspace::base) fn generation(&self) -> u64 {
        match &self.pending {
            PreparedPendingChanges::Mutation(view) => view.generation(),
            PreparedPendingChanges::Complete(delta) => delta.generation(),
        }
    }

    fn len(&self) -> usize {
        match &self.pending {
            PreparedPendingChanges::Mutation(view) => view.len(),
            PreparedPendingChanges::Complete(delta) => delta.changes().len(),
        }
    }

    pub(in crate::workspace::base) fn change(&self, index: usize) -> Option<&WorkspaceChange> {
        match &self.pending {
            PreparedPendingChanges::Mutation(view) => view.get(index),
            PreparedPendingChanges::Complete(delta) => delta.changes().get(index),
        }
    }

    fn new_action(&self, name: &str) -> Option<&WorkspaceChange> {
        match &self.pending {
            PreparedPendingChanges::Mutation(view) => {
                Some(view.prepared_change()).filter(|change| change.name() == Some(name))
            }
            PreparedPendingChanges::Complete(delta) => delta.changes()[self.source_start..]
                .binary_search_by(|change| change.name().expect("named completion").cmp(name))
                .ok()
                .and_then(|index| delta.changes().get(self.source_start + index)),
        }
    }

    pub(in crate::workspace::base) fn global_target(
        &self,
        name: &str,
    ) -> Option<&GlobalBindingTarget> {
        if matches!(self.pending, PreparedPendingChanges::Mutation(_)) {
            if matches!(
                self.pending,
                PreparedPendingChanges::Mutation(ref view)
                    if matches!(view.prepared_change().data(), WorkspaceChangeData::Clear)
            ) {
                return None;
            }
            return match self.new_action(name).map(WorkspaceChange::data) {
                Some(WorkspaceChangeData::BindGlobal { target, .. }) => Some(target),
                Some(WorkspaceChangeData::Assign { .. })
                | Some(WorkspaceChangeData::UnbindGlobal { .. })
                | Some(WorkspaceChangeData::Clear) => None,
                Some(WorkspaceChangeData::Remove { .. }) | None => self
                    .managed_storage
                    .expect("prepared live mutation")
                    .global_target(name)
                    .or_else(|| {
                        (!self.captures_legacy)
                            .then(|| self.base.global_target(name))
                            .flatten()
                    }),
            };
        }
        if let Some(globals) = self.globals {
            return globals.get(name);
        }
        let managed_storage = self.managed_storage.expect("prepared completion storage");
        if let Some(global) = managed_storage.global(name) {
            return global
                .then(|| managed_storage.global_target(name))
                .flatten();
        }
        if let Some(action) = self.new_action(name) {
            match action.data() {
                WorkspaceChangeData::BindGlobal { target, .. } => return Some(target),
                WorkspaceChangeData::Assign { .. }
                | WorkspaceChangeData::UnbindGlobal { .. }
                | WorkspaceChangeData::Clear => return None,
                WorkspaceChangeData::Remove { .. } => {}
            }
        }
        if self.captures_legacy {
            None
        } else {
            self.base.global_target(name)
        }
    }

    pub(in crate::workspace::base) fn lookup(&self, name: &str) -> Option<&Value> {
        if self.global_target(name).is_some() {
            return None;
        }
        if matches!(self.pending, PreparedPendingChanges::Mutation(_)) {
            if matches!(
                self.pending,
                PreparedPendingChanges::Mutation(ref view)
                    if matches!(view.prepared_change().data(), WorkspaceChangeData::Clear)
            ) {
                return None;
            }
            return self.new_action(name).map_or_else(
                || {
                    self.managed_storage
                        .expect("prepared live mutation")
                        .lookup(name)
                        .or_else(|| {
                            (!self.captures_legacy)
                                .then(|| self.base.lookup(name))
                                .flatten()
                        })
                },
                WorkspaceChange::value,
            );
        }
        if let Some(entries) = self.entries {
            return entries.get(name).map(RootedWorkspaceEntry::value);
        }
        if let Some(value) = self
            .managed_storage
            .expect("prepared completion storage")
            .lookup(name)
        {
            return Some(value);
        }
        if let Some(action) = self.new_action(name) {
            return action.value();
        }
        if self.captures_legacy {
            None
        } else {
            self.base.lookup(name)
        }
    }
}

impl PreparedProjectionMutation {
    pub(in crate::workspace::base) fn publication_view<'a>(
        &'a self,
        base: &'a BaseWorkspaceProjection,
    ) -> Result<PreparedProjectionPublicationView<'a>, BaseWorkspaceError> {
        let base = base.ordered_view()?;
        let pending = PreparedPendingChanges::Mutation(PreparedWorkspaceDeltaView::new(
            base.pending(),
            &self.pending,
        ));
        Ok(PreparedProjectionPublicationView {
            base,
            pending,
            managed_storage: Some(&self.live),
            entries: None,
            globals: None,
            captures_legacy: self.captures_legacy,
            source_start: 0,
            expected_revision: self.expected_revision,
            expected_generation: self.expected_generation,
        })
    }
}

impl PreparedProjectionCompletion {
    pub(in crate::workspace::base) fn publication_view<'a>(
        &'a self,
        base: &'a BaseWorkspaceProjection,
    ) -> Result<PreparedProjectionPublicationView<'a>, BaseWorkspaceError> {
        Ok(PreparedProjectionPublicationView {
            base: base.ordered_view()?,
            pending: PreparedPendingChanges::Complete(&self.pending),
            managed_storage: Some(&self.managed_storage),
            entries: self.entries.as_ref(),
            globals: self.global_bindings.as_ref(),
            captures_legacy: self.captures_legacy,
            source_start: self.source_start,
            expected_revision: self.expected_revision,
            expected_generation: self.expected_generation,
        })
    }
}
