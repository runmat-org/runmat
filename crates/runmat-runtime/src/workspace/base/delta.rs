use runmat_value::Value;
use std::alloc::Layout;

use super::journal::{PreparedWorkspaceImpacts, WorkspaceImpact};
use super::values::{RootedWorkspaceEntry, WorkspaceEntryName};
use super::BaseWorkspaceError;
use crate::workspace::GlobalBindingTarget;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceChangeKind {
    Assign,
    Remove,
    Clear,
}

#[derive(Clone, Debug)]
pub(super) enum WorkspaceChangeData {
    Assign {
        name: WorkspaceEntryName,
        entry: RootedWorkspaceEntry,
        executor_slot: Option<usize>,
    },
    Remove {
        name: WorkspaceEntryName,
    },
    BindGlobal {
        name: WorkspaceEntryName,
        target: GlobalBindingTarget,
    },
    UnbindGlobal {
        name: WorkspaceEntryName,
    },
    Clear,
}

/// Whether the authentic producer has already staged this change in the
/// canonical managed overlay. This is storage provenance, independent of name
/// representation or whether a Value happens to be admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WorkspaceChangeStorage {
    Legacy,
    ManagedOverlay,
}

/// One checked workspace mutation. Its representation is private so an
/// assigning change cannot carry an unrooted or transient value.
#[derive(Clone, Debug)]
pub struct WorkspaceChange(WorkspaceChangeData, WorkspaceChangeStorage);

impl WorkspaceChange {
    pub fn assign(name: impl Into<String>, value: Value) -> Result<Self, BaseWorkspaceError> {
        Ok(Self::assign_entry(
            WorkspaceEntryName::from(name.into()),
            RootedWorkspaceEntry::new(value)?,
        ))
    }

    pub fn remove(name: impl Into<String>) -> Self {
        Self(
            WorkspaceChangeData::Remove {
                name: WorkspaceEntryName::from(name.into()),
            },
            WorkspaceChangeStorage::Legacy,
        )
    }

    pub const fn clear() -> Self {
        Self(WorkspaceChangeData::Clear, WorkspaceChangeStorage::Legacy)
    }

    pub fn kind(&self) -> WorkspaceChangeKind {
        match &self.0 {
            WorkspaceChangeData::Assign { .. } => WorkspaceChangeKind::Assign,
            WorkspaceChangeData::Remove { .. }
            | WorkspaceChangeData::BindGlobal { .. }
            | WorkspaceChangeData::UnbindGlobal { .. } => WorkspaceChangeKind::Remove,
            WorkspaceChangeData::Clear => WorkspaceChangeKind::Clear,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match &self.0 {
            WorkspaceChangeData::Assign { name, .. }
            | WorkspaceChangeData::Remove { name }
            | WorkspaceChangeData::BindGlobal { name, .. }
            | WorkspaceChangeData::UnbindGlobal { name } => Some(name.as_str()),
            WorkspaceChangeData::Clear => None,
        }
    }

    pub(super) fn entry_name(&self) -> Option<&WorkspaceEntryName> {
        match &self.0 {
            WorkspaceChangeData::Assign { name, .. }
            | WorkspaceChangeData::Remove { name }
            | WorkspaceChangeData::BindGlobal { name, .. }
            | WorkspaceChangeData::UnbindGlobal { name } => Some(name),
            WorkspaceChangeData::Clear => None,
        }
    }

    pub fn value(&self) -> Option<&Value> {
        match &self.0 {
            WorkspaceChangeData::Assign { entry, .. } => Some(entry.value()),
            WorkspaceChangeData::Remove { .. }
            | WorkspaceChangeData::BindGlobal { .. }
            | WorkspaceChangeData::UnbindGlobal { .. }
            | WorkspaceChangeData::Clear => None,
        }
    }

    pub(super) fn assign_entry(
        name: impl Into<WorkspaceEntryName>,
        entry: RootedWorkspaceEntry,
    ) -> Self {
        Self(
            WorkspaceChangeData::Assign {
                name: name.into(),
                entry,
                executor_slot: None,
            },
            WorkspaceChangeStorage::Legacy,
        )
    }

    pub(super) fn assign_executor_entry(
        name: WorkspaceEntryName,
        entry: RootedWorkspaceEntry,
        slot: usize,
    ) -> Self {
        Self(
            WorkspaceChangeData::Assign {
                name,
                entry,
                executor_slot: Some(slot),
            },
            WorkspaceChangeStorage::Legacy,
        )
    }

    pub(super) fn remove_prepared(name: WorkspaceEntryName) -> Self {
        Self(
            WorkspaceChangeData::Remove { name },
            WorkspaceChangeStorage::Legacy,
        )
    }

    pub(super) fn bind_global_prepared(
        name: WorkspaceEntryName,
        target: GlobalBindingTarget,
    ) -> Self {
        Self(
            WorkspaceChangeData::BindGlobal { name, target },
            WorkspaceChangeStorage::Legacy,
        )
    }

    pub(super) fn unbind_global_prepared(name: WorkspaceEntryName) -> Self {
        Self(
            WorkspaceChangeData::UnbindGlobal { name },
            WorkspaceChangeStorage::Legacy,
        )
    }

    pub(super) fn staged_in_managed_overlay(mut self) -> Self {
        self.1 = WorkspaceChangeStorage::ManagedOverlay;
        self
    }

    pub(super) fn with_storage(mut self, storage: WorkspaceChangeStorage) -> Self {
        self.1 = storage;
        self
    }

    pub(super) fn mark_staged_in_managed_overlay(&mut self) {
        self.1 = WorkspaceChangeStorage::ManagedOverlay;
    }

    pub(super) fn storage(&self) -> WorkspaceChangeStorage {
        self.1
    }

    pub(super) fn executor_slot(&self) -> Option<usize> {
        match &self.0 {
            WorkspaceChangeData::Assign { executor_slot, .. } => *executor_slot,
            _ => None,
        }
    }

    pub(super) fn data(&self) -> &WorkspaceChangeData {
        &self.0
    }

    fn is_allocation_free_clone(&self) -> bool {
        let prepared_name = self
            .entry_name()
            .is_none_or(|name| matches!(name, WorkspaceEntryName::Prepared(_)));
        prepared_name
            && match self.data() {
                WorkspaceChangeData::Assign { entry, .. } => entry.is_prepared(),
                _ => true,
            }
    }
}

/// Ordered, unpublished workspace mutations.
///
/// Every explicitly ordered assigned value remains rooted until the delta is published or
/// dropped, including an assignment later overwritten, removed, or cleared in
/// the same delta. This is intentional: the pending record is replayable in
/// order. A successful commit returns a canonical-effect delta which omits
/// intermediate values that no longer contribute to workspace state. The
/// internal executor-projection entry point separately coalesces superseded
/// stores within its uninterrupted suffix; it does not change this public API.
#[derive(Debug)]
pub struct WorkspaceDelta {
    changes: Vec<WorkspaceChange>,
    executor_suffix_start: usize,
    generation: u64,
    // The changes Vec is declared first so its backing retires before the
    // exact claim refunds that physical allocation.
    _admitted_backing: Option<runmat_value::ExactHostAllocationClaim>,
    prepared_impacts: PreparedWorkspaceImpacts,
    prepared_impacts_complete: bool,
    has_prepared_mutation: bool,
}

impl Default for WorkspaceDelta {
    fn default() -> Self {
        Self::new()
    }
}

pub(super) enum PreparedWorkspaceDeltaMutation {
    Replace {
        index: usize,
        assignment: WorkspaceChange,
        impacts: PreparedWorkspaceImpacts,
        after_generation: u64,
    },
    ReplaceDelta(WorkspaceDelta),
}

/// Prior rooted entries are carried out of a projection borrow instead of
/// invoking their final Foreign release while the workspace is borrowed.
pub(super) enum RetiredWorkspaceDeltaMutation {
    Change { _change: WorkspaceChange },
    Delta { _delta: WorkspaceDelta },
}

/// One or two actual inline changes admitted by the explicit unbounded
/// policy. The generation is checked before the observer sees the transition;
/// installation consumes these same changes, not a replay or cloned delta.
pub(super) struct PreparedUnboundedDeltaTransition {
    before_generation: u64,
    after_generation: u64,
    changes: [Option<WorkspaceChange>; 2],
    coalesce_assignment: bool,
    origin: WorkspaceMutationOrigin,
    clear_delta: Option<WorkspaceDelta>,
}

/// Public ordered changes preserve replay history. Executor effects may
/// retire only the already-coalescible internal suffix.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::workspace::base) enum WorkspaceMutationOrigin {
    Ordered,
    Executor,
}

impl PreparedUnboundedDeltaTransition {
    pub(super) fn before_generation(&self) -> u64 {
        self.before_generation
    }

    pub(super) fn after_generation(&self) -> u64 {
        self.after_generation
    }

    fn appended_slots(&self, current: &WorkspaceDelta) -> usize {
        if self.changes[1].is_some() {
            return 2;
        }
        if !self.coalesce_assignment {
            return 1;
        }
        let first = self.changes[0]
            .as_ref()
            .expect("prepared transition change");
        let WorkspaceChangeData::Assign { name, .. } = first.data() else {
            return 1;
        };
        usize::from(
            current
                .executor_assignment_replacement(name.as_str())
                .is_none(),
        )
    }
}

/// The exact pending changes after an admitted mutation, before installing it
/// in the live projection. Replacing an executor assignment borrows the old
/// backing and substitutes only that one prepared change.
pub(super) struct PreparedWorkspaceDeltaView<'a> {
    current: &'a WorkspaceDelta,
    prepared: &'a PreparedWorkspaceDeltaMutation,
}

impl<'a> PreparedWorkspaceDeltaView<'a> {
    pub(super) fn new(
        current: &'a WorkspaceDelta,
        prepared: &'a PreparedWorkspaceDeltaMutation,
    ) -> Self {
        Self { current, prepared }
    }

    pub(super) fn len(&self) -> usize {
        match self.prepared {
            PreparedWorkspaceDeltaMutation::Replace { .. } => self.current.changes.len(),
            PreparedWorkspaceDeltaMutation::ReplaceDelta(delta) => delta.changes.len(),
        }
    }

    pub(super) fn get(&self, index: usize) -> Option<&WorkspaceChange> {
        if index >= self.len() {
            return None;
        }
        Some(match self.prepared {
            PreparedWorkspaceDeltaMutation::Replace {
                index: replaced,
                assignment,
                ..
            } if index == *replaced => assignment,
            PreparedWorkspaceDeltaMutation::Replace { .. } => &self.current.changes[index],
            PreparedWorkspaceDeltaMutation::ReplaceDelta(delta) => &delta.changes[index],
        })
    }

    pub(super) fn generation(&self) -> u64 {
        match self.prepared {
            PreparedWorkspaceDeltaMutation::Replace {
                after_generation, ..
            } => *after_generation,
            PreparedWorkspaceDeltaMutation::ReplaceDelta(delta) => delta.generation,
        }
    }

    pub(super) fn prepared_change(&self) -> &WorkspaceChange {
        match self.prepared {
            PreparedWorkspaceDeltaMutation::Replace { assignment, .. } => assignment,
            PreparedWorkspaceDeltaMutation::ReplaceDelta(delta) => delta
                .changes
                .last()
                .expect("prepared mutation has a final change"),
        }
    }
}

pub(super) struct PreparedWorkspaceDeltaBuilder {
    changes: Vec<WorkspaceChange>,
    claim: runmat_value::ExactHostAllocationClaim,
    prepared_impacts: PreparedWorkspaceImpacts,
    executor_suffix_start: usize,
    generation: u64,
    expected_len: usize,
}

impl PreparedWorkspaceDeltaBuilder {
    pub(super) fn push(
        &mut self,
        change: WorkspaceChange,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<(), crate::context::host_allocation::HostAllocationError> {
        if self.changes.len() >= self.expected_len {
            return Err(crate::context::host_allocation::HostAllocationError::Overflow);
        }
        let impact = match change.data() {
            WorkspaceChangeData::Assign { name, .. }
            | WorkspaceChangeData::Remove { name }
            | WorkspaceChangeData::BindGlobal { name, .. }
            | WorkspaceChangeData::UnbindGlobal { name } => WorkspaceImpact::Name(name.clone()),
            WorkspaceChangeData::Clear => WorkspaceImpact::Clear,
        };
        self.prepared_impacts = self.prepared_impacts.push(impact, permit)?;
        self.changes.push(change);
        Ok(())
    }

    pub(super) fn finish(
        self,
    ) -> Result<WorkspaceDelta, crate::context::host_allocation::HostAllocationError> {
        if self.changes.len() != self.expected_len {
            return Err(crate::context::host_allocation::HostAllocationError::Overflow);
        }
        Ok(WorkspaceDelta {
            changes: self.changes,
            executor_suffix_start: self.executor_suffix_start,
            generation: self.generation,
            _admitted_backing: Some(self.claim),
            prepared_impacts: self.prepared_impacts,
            prepared_impacts_complete: true,
            has_prepared_mutation: true,
        })
    }
}

impl WorkspaceDelta {
    pub(super) fn independent_effect_retained_bytes(
        capacity: usize,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let raw = Layout::array::<WorkspaceChange>(capacity)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw)
                .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?,
            1,
        )
    }

    pub(super) fn prepare_independent_effect(
        &self,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        let raw = Layout::array::<WorkspaceChange>(self.changes.len())
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let claim = permit.reserve_exact_claim(raw)?;
        let mut changes = runmat_value::exact_capacity_vec(self.changes.len())?;
        changes.extend(self.changes.iter().cloned());
        Ok(Self::from_admitted_changes(changes, claim))
    }

    pub(super) fn into_effect_backing(
        mut self,
    ) -> (
        Vec<WorkspaceChange>,
        Option<runmat_value::ExactHostAllocationClaim>,
    ) {
        (
            std::mem::take(&mut self.changes),
            self._admitted_backing.take(),
        )
    }

    pub const fn new() -> Self {
        Self {
            changes: Vec::new(),
            executor_suffix_start: 0,
            generation: 0,
            _admitted_backing: None,
            prepared_impacts: PreparedWorkspaceImpacts::new(),
            prepared_impacts_complete: true,
            has_prepared_mutation: false,
        }
    }

    pub fn from_changes(changes: Vec<WorkspaceChange>) -> Self {
        Self {
            executor_suffix_start: changes.len(),
            changes,
            generation: 0,
            _admitted_backing: None,
            prepared_impacts: PreparedWorkspaceImpacts::default(),
            prepared_impacts_complete: false,
            has_prepared_mutation: false,
        }
    }

    pub(super) fn from_admitted_changes(
        changes: Vec<WorkspaceChange>,
        claim: runmat_value::ExactHostAllocationClaim,
    ) -> Self {
        Self {
            executor_suffix_start: changes.len(),
            changes,
            generation: 0,
            _admitted_backing: Some(claim),
            prepared_impacts: PreparedWorkspaceImpacts::default(),
            prepared_impacts_complete: false,
            has_prepared_mutation: false,
        }
    }

    pub fn assign(
        &mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<(), BaseWorkspaceError> {
        self.prepare_legacy_mutation();
        self.changes.push(WorkspaceChange::assign(name, value)?);
        self.executor_suffix_start = self.changes.len();
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    pub fn remove(&mut self, name: impl Into<String>) {
        self.prepare_legacy_mutation();
        self.changes.push(WorkspaceChange::remove(name));
        self.executor_suffix_start = self.changes.len();
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        self.prepare_legacy_mutation();
        self.changes.push(WorkspaceChange::clear());
        self.executor_suffix_start = self.changes.len();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Bind a base-workspace name to the separate session-global value store.
    /// The canonical base owner records this alongside removing any local
    /// value so both namespace changes share one ordered revision.
    pub fn bind_global(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.bind_global_target(
            name.clone(),
            GlobalBindingTarget::Named(crate::workspace::WorkspaceName::unbounded(&name)),
        );
    }

    pub fn bind_global_target(&mut self, name: impl Into<String>, target: GlobalBindingTarget) {
        self.prepare_legacy_mutation();
        self.changes.push(WorkspaceChange(
            WorkspaceChangeData::BindGlobal {
                name: WorkspaceEntryName::from(name.into()),
                target,
            },
            WorkspaceChangeStorage::Legacy,
        ));
        self.executor_suffix_start = self.changes.len();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Remove a base-workspace global binding and any hidden local value.
    /// A later local assignment must be explicit; stale projections cannot
    /// leave a value that resurfaces when the binding is cleared.
    pub fn unbind_global(&mut self, name: impl Into<String>) {
        self.prepare_legacy_mutation();
        self.changes.push(WorkspaceChange(
            WorkspaceChangeData::UnbindGlobal {
                name: WorkspaceEntryName::from(name.into()),
            },
            WorkspaceChangeStorage::Legacy,
        ));
        self.executor_suffix_start = self.changes.len();
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.changes.len()
    }

    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn completion_retained_bytes<'a>(
        &self,
        additional: usize,
        names: impl Iterator<Item = &'a str>,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        if self
            .changes
            .iter()
            .any(|change| !change.is_allocation_free_clone())
        {
            return Err(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            );
        }
        let len = self
            .changes
            .len()
            .checked_add(additional)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        let raw = Layout::array::<WorkspaceChange>(len)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let mut retained =
            crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
                u64::try_from(raw)
                    .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?,
                1,
            )?;
        for name in names {
            retained = retained
                .checked_add(self.prepared_impacts.retained_bytes_for_name(name)?)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        }
        Ok(retained)
    }

    pub(super) fn prepare_completion(
        &self,
        additional: usize,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<PreparedWorkspaceDeltaBuilder, crate::context::host_allocation::HostAllocationError>
    {
        if self
            .changes
            .iter()
            .any(|change| !change.is_allocation_free_clone())
        {
            return Err(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            );
        }
        let expected_len = self
            .changes
            .len()
            .checked_add(additional)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        let raw = Layout::array::<WorkspaceChange>(expected_len)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let claim = permit.reserve_exact_claim(raw)?;
        let mut changes = runmat_value::exact_capacity_vec(expected_len)?;
        changes.extend(self.changes.iter().cloned());
        Ok(PreparedWorkspaceDeltaBuilder {
            changes,
            claim,
            prepared_impacts: self.prepared_impacts.clone(),
            executor_suffix_start: self.executor_suffix_start,
            generation: self
                .generation
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?,
            expected_len,
        })
    }

    pub fn changes(&self) -> &[WorkspaceChange] {
        &self.changes
    }

    pub(super) fn changes_mut(&mut self) -> &mut [WorkspaceChange] {
        &mut self.changes
    }

    pub(super) fn is_prepared_assignments_only(&self) -> bool {
        !self.changes.is_empty()
            && self.changes.iter().all(|change| {
                matches!(
                    change.data(),
                    WorkspaceChangeData::Assign {
                        entry,
                        ..
                    } if entry.is_prepared()
                )
            })
    }

    /// Collapse this admitted ordered delta to the canonical final effect
    /// without allocating another name set or change buffer. The caller has
    /// already applied the ordered mutations to canonical state; `state`
    /// resolves its final local value and independent global target.
    pub(super) fn canonicalize_effect(
        &mut self,
        mut state: impl FnMut(
            &WorkspaceEntryName,
        ) -> (
            Option<RootedWorkspaceEntry>,
            Option<GlobalBindingTarget>,
            WorkspaceChangeStorage,
        ),
    ) {
        let last_clear = self
            .changes
            .iter()
            .rposition(|change| matches!(change.data(), WorkspaceChangeData::Clear));
        if let Some(index) = last_clear {
            self.changes.drain(..index);
        }
        let named_start = usize::from(last_clear.is_some());

        self.changes[named_start..].sort_unstable_by(|left, right| {
            left.entry_name()
                .expect("canonical named effect")
                .cmp(right.entry_name().expect("canonical named effect"))
        });
        let mut index = named_start;
        while index < self.changes.len() {
            let name = self.changes[index]
                .entry_name()
                .expect("canonical named effect");
            let mut end = index + 1;
            while end < self.changes.len() && self.changes[end].entry_name() == Some(name) {
                end += 1;
            }
            let changed_namespace = self.changes[index..end].iter().any(|change| {
                matches!(
                    change.data(),
                    WorkspaceChangeData::BindGlobal { .. }
                        | WorkspaceChangeData::UnbindGlobal { .. }
                )
            });
            let (local, global, storage) = state(name);
            let name = match &mut self.changes[index].0 {
                WorkspaceChangeData::Assign { name, .. }
                | WorkspaceChangeData::Remove { name }
                | WorkspaceChangeData::BindGlobal { name, .. }
                | WorkspaceChangeData::UnbindGlobal { name } => {
                    std::mem::replace(name, WorkspaceEntryName::Owned(String::new()))
                }
                WorkspaceChangeData::Clear => unreachable!("named canonical effect"),
            };
            self.changes[index] = match (local, global) {
                (_, Some(target)) => {
                    WorkspaceChange(WorkspaceChangeData::BindGlobal { name, target }, storage)
                }
                (Some(entry), None) => WorkspaceChange(
                    WorkspaceChangeData::Assign {
                        name,
                        entry,
                        executor_slot: None,
                    },
                    storage,
                ),
                (None, None) if changed_namespace => WorkspaceChange(
                    WorkspaceChangeData::UnbindGlobal { name },
                    WorkspaceChangeStorage::Legacy,
                ),
                (None, None) => WorkspaceChange(
                    WorkspaceChangeData::Remove { name },
                    WorkspaceChangeStorage::Legacy,
                ),
            };
            index = end;
        }
        self.changes.dedup_by(|later, earlier| {
            later.entry_name().is_some() && later.entry_name() == earlier.entry_name()
        });
        if last_clear.is_some() {
            self.changes.retain(|change| {
                !matches!(
                    change.data(),
                    WorkspaceChangeData::Remove { .. } | WorkspaceChangeData::UnbindGlobal { .. }
                )
            });
        }
        self.executor_suffix_start = self.changes.len();
    }

    pub(super) fn push_reconciled_change(&mut self, change: WorkspaceChange) {
        self.prepare_legacy_mutation();
        self.changes.push(change);
        self.executor_suffix_start = self.changes.len();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Clone this pending transaction under the explicit unbounded host
    /// policy. Prepared names, values, and impact nodes retain their existing
    /// immutable owners; only the ordered change buffer is copied.
    pub(super) fn clone_unbounded(&self) -> Self {
        Self {
            changes: self.changes.clone(),
            executor_suffix_start: self.executor_suffix_start,
            generation: self.generation,
            _admitted_backing: None,
            prepared_impacts: self.prepared_impacts.clone(),
            prepared_impacts_complete: self.prepared_impacts_complete,
            has_prepared_mutation: self.has_prepared_mutation,
        }
    }

    /// The caller staged this exact detached action sequence in the managed
    /// overlay before handing the delta to publication.
    pub(super) fn mark_detachment_overlay_backed(&mut self) {
        for change in &mut self.changes {
            change.1 = WorkspaceChangeStorage::ManagedOverlay;
        }
    }

    pub(super) fn prepare_unbounded_transition(
        &self,
        first: WorkspaceChange,
        second: Option<WorkspaceChange>,
        coalesce_assignment: bool,
        origin: WorkspaceMutationOrigin,
    ) -> Result<
        PreparedUnboundedDeltaTransition,
        crate::context::host_allocation::HostAllocationError,
    > {
        let count = u64::from(second.is_some()) + 1;
        let after_generation = self
            .generation
            .checked_add(count)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        let clear_delta = if origin == WorkspaceMutationOrigin::Executor
            && matches!(first.data(), WorkspaceChangeData::Clear)
            && second.is_none()
        {
            let len = self
                .executor_suffix_start
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
            let changes = runmat_value::exact_capacity_vec_with(len, |index| {
                if index + 1 == len {
                    first.clone()
                } else {
                    self.changes[index].clone()
                }
            })?;
            Some(Self {
                changes,
                executor_suffix_start: self.executor_suffix_start,
                generation: after_generation,
                _admitted_backing: None,
                prepared_impacts: self.prepared_impacts.clone(),
                prepared_impacts_complete: self.prepared_impacts_complete,
                has_prepared_mutation: self.has_prepared_mutation,
            })
        } else {
            None
        };
        Ok(PreparedUnboundedDeltaTransition {
            before_generation: self.generation,
            after_generation,
            changes: [Some(first), second],
            coalesce_assignment,
            origin,
            clear_delta,
        })
    }

    pub(super) fn reserve_unbounded_transition(
        &mut self,
        transition: &PreparedUnboundedDeltaTransition,
    ) -> Result<(), crate::context::host_allocation::HostAllocationError> {
        assert_eq!(self.generation, transition.before_generation);
        let additional = transition.appended_slots(self);
        self.prepare_legacy_mutation();
        self.changes.try_reserve(additional).map_err(|_| {
            crate::context::host_allocation::HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            )
        })
    }

    pub(super) fn install_unbounded_transition(
        &mut self,
        mut transition: PreparedUnboundedDeltaTransition,
    ) -> ([Option<WorkspaceChange>; 2], Option<Self>) {
        assert_eq!(self.generation, transition.before_generation);
        if let Some(candidate) = transition.clear_delta.take() {
            return ([None, None], Some(std::mem::replace(self, candidate)));
        }
        let mut retired = [None, None];
        for (retired_slot, change) in retired.iter_mut().zip(&mut transition.changes) {
            let Some(change) = change.take() else {
                continue;
            };
            self.prepare_legacy_mutation();
            let assignment = matches!(change.data(), WorkspaceChangeData::Assign { .. });
            if !assignment && transition.origin == WorkspaceMutationOrigin::Executor {
                // Executor namespace changes retire a superseded internal
                // store, but retain namespace ordering and the authentic
                // replay prefix. Return the displaced owner to the caller.
                if let Some(name) = change.name() {
                    if let Some(index) =
                        self.changes[self.executor_suffix_start..]
                            .iter()
                            .position(|old| {
                                matches!(old.data(), WorkspaceChangeData::Assign { .. })
                                    && old.name() == Some(name)
                            })
                    {
                        *retired_slot =
                            Some(self.changes.remove(self.executor_suffix_start + index));
                    }
                }
            }
            if assignment && transition.coalesce_assignment {
                let name = change.name().expect("named prepared assignment");
                if let Some(previous) = self.changes[self.executor_suffix_start..]
                    .iter_mut()
                    .rev()
                    .find(|previous| {
                        matches!(previous.data(), WorkspaceChangeData::Assign { .. })
                            && previous.name() == Some(name)
                    })
                {
                    *retired_slot = Some(std::mem::replace(previous, change));
                    self.generation += 1;
                    continue;
                }
            }
            self.changes.push(change);
            if transition.origin == WorkspaceMutationOrigin::Ordered {
                self.executor_suffix_start = self.changes.len();
            }
            self.generation += 1;
        }
        assert_eq!(self.generation, transition.after_generation);
        (retired, None)
    }

    /// Detachment actions name independent bindings. Canonicalize their
    /// order before one prospective completion; repeated selected names have
    /// the same removal action and need only one published effect.
    pub(super) fn normalize_detachment_names(&mut self) {
        let start = usize::from(
            self.changes
                .first()
                .is_some_and(|change| change.kind() == WorkspaceChangeKind::Clear),
        );
        self.changes[start..].sort_unstable_by(|left, right| left.name().cmp(&right.name()));
        // A leading Clear has no name and cannot equal any named removal.
        // Deduplicate the Vec itself so truncation retires duplicate owners.
        self.changes
            .dedup_by(|left, right| left.name() == right.name());
        self.executor_suffix_start = self.changes.len();
    }

    pub(super) fn executor_assignment_retained_bytes(
        &self,
        name: &str,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let impact_bytes = if self.prepared_impacts_complete {
            self.prepared_impacts.retained_bytes_for_name(name)?
        } else {
            0
        };
        if self.executor_assignment_replacement(name).is_some() {
            return Ok(impact_bytes);
        }
        let len = self.executor_assignment_len(name)?;
        let raw = Layout::array::<WorkspaceChange>(len)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let changes =
            crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
                u64::try_from(raw)
                    .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?,
                1,
            )?;
        changes
            .checked_add(impact_bytes)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)
    }

    pub(super) fn prepare_executor_assignment(
        &self,
        name: WorkspaceEntryName,
        entry: RootedWorkspaceEntry,
        slot: Option<usize>,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<PreparedWorkspaceDeltaMutation, crate::context::host_allocation::HostAllocationError>
    {
        let after_generation = self
            .generation
            .checked_add(1)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        let replacement = self.executor_assignment_replacement(name.as_str());
        let len = self.executor_assignment_len(name.as_str())?;
        let impacts = if self.prepared_impacts_complete {
            self.prepared_impacts
                .push(WorkspaceImpact::Name(name.clone()), permit)?
        } else {
            PreparedWorkspaceImpacts::default()
        };
        let assignment = match slot {
            Some(slot) => WorkspaceChange::assign_executor_entry(name, entry, slot),
            None => WorkspaceChange::assign_entry(name, entry),
        }
        .staged_in_managed_overlay();
        if let Some(index) = replacement {
            return Ok(PreparedWorkspaceDeltaMutation::Replace {
                index,
                assignment,
                impacts,
                after_generation,
            });
        }
        let raw = Layout::array::<WorkspaceChange>(len)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let claim = permit.reserve_exact_claim(raw)?;
        let changes = runmat_value::exact_capacity_vec_with(len, |index| {
            if index + 1 == len {
                assignment.clone()
            } else {
                self.changes[index].clone()
            }
        })?;
        Ok(PreparedWorkspaceDeltaMutation::ReplaceDelta(Self {
            changes,
            executor_suffix_start: self.executor_suffix_start,
            generation: after_generation,
            _admitted_backing: Some(claim),
            prepared_impacts: impacts,
            prepared_impacts_complete: self.prepared_impacts_complete,
            has_prepared_mutation: true,
        }))
    }

    pub(super) fn apply_prepared_mutation(
        &mut self,
        mutation: PreparedWorkspaceDeltaMutation,
    ) -> RetiredWorkspaceDeltaMutation {
        match mutation {
            PreparedWorkspaceDeltaMutation::Replace {
                index,
                assignment,
                impacts,
                after_generation,
            } => {
                let retired = std::mem::replace(&mut self.changes[index], assignment);
                self.prepared_impacts = impacts;
                self.has_prepared_mutation = true;
                self.generation = after_generation;
                RetiredWorkspaceDeltaMutation::Change { _change: retired }
            }
            PreparedWorkspaceDeltaMutation::ReplaceDelta(delta) => {
                RetiredWorkspaceDeltaMutation::Delta {
                    _delta: std::mem::replace(self, delta),
                }
            }
        }
    }

    pub(super) fn namespace_mutation_retained_bytes(
        &self,
        name: &str,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let impact_bytes = if self.prepared_impacts_complete {
            self.prepared_impacts.retained_bytes_for_name(name)?
        } else {
            0
        };
        self.appended_change_retained_bytes(impact_bytes)
    }

    pub(super) fn prepare_remove(
        &self,
        name: WorkspaceEntryName,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        self.prepare_appended_change(
            WorkspaceChange(
                WorkspaceChangeData::Remove { name },
                WorkspaceChangeStorage::ManagedOverlay,
            ),
            permit,
        )
    }

    pub(super) fn prepare_bind_global(
        &self,
        name: WorkspaceEntryName,
        target: GlobalBindingTarget,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        self.prepare_appended_change(
            WorkspaceChange(
                WorkspaceChangeData::BindGlobal { name, target },
                WorkspaceChangeStorage::ManagedOverlay,
            ),
            permit,
        )
    }

    pub(super) fn prepare_unbind_global(
        &self,
        name: WorkspaceEntryName,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        self.prepare_appended_change(
            WorkspaceChange(
                WorkspaceChangeData::UnbindGlobal { name },
                WorkspaceChangeStorage::ManagedOverlay,
            ),
            permit,
        )
    }

    pub(super) fn clear_retained_bytes(
        &self,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let impact_bytes = if self.prepared_impacts_complete {
            self.prepared_impacts.retained_bytes_for_clear()?
        } else {
            0
        };
        let len = self
            .executor_suffix_start
            .checked_add(1)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        Self::change_retained_bytes(len, impact_bytes)
    }

    pub(super) fn prepare_clear(
        &self,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        // A clear supersedes the coalesced executor suffix, just as a later
        // source store supersedes an earlier source store. Explicitly ordered
        // changes before that suffix retain their public replay semantics.
        self.prepare_change_after_prefix(
            WorkspaceChange::clear().staged_in_managed_overlay(),
            self.executor_suffix_start,
            permit,
        )
    }

    fn appended_change_retained_bytes(
        &self,
        impact_bytes: u64,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let len = self
            .changes
            .len()
            .checked_add(1)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        Self::change_retained_bytes(len, impact_bytes)
    }

    fn change_retained_bytes(
        len: usize,
        impact_bytes: u64,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let raw = Layout::array::<WorkspaceChange>(len)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let changes =
            crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
                u64::try_from(raw)
                    .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?,
                1,
            )?;
        changes
            .checked_add(impact_bytes)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)
    }

    fn prepare_appended_change(
        &self,
        appended: WorkspaceChange,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        self.prepare_change_after_prefix(appended, self.changes.len(), permit)
    }

    fn prepare_change_after_prefix(
        &self,
        appended: WorkspaceChange,
        prefix_len: usize,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, crate::context::host_allocation::HostAllocationError> {
        let impact = match appended.data() {
            WorkspaceChangeData::Assign { name, .. }
            | WorkspaceChangeData::Remove { name }
            | WorkspaceChangeData::BindGlobal { name, .. }
            | WorkspaceChangeData::UnbindGlobal { name } => WorkspaceImpact::Name(name.clone()),
            WorkspaceChangeData::Clear => WorkspaceImpact::Clear,
        };
        let prepared_impacts = if self.prepared_impacts_complete {
            self.prepared_impacts.push(impact, permit)?
        } else {
            PreparedWorkspaceImpacts::default()
        };
        let replaced = appended.name().and_then(|name| {
            self.changes[self.executor_suffix_start..prefix_len]
                .iter()
                .position(|old| {
                    matches!(old.data(), WorkspaceChangeData::Assign { .. })
                        && old.name() == Some(name)
                })
                .map(|index| self.executor_suffix_start + index)
        });
        let len = prefix_len
            .saturating_sub(usize::from(replaced.is_some()))
            .checked_add(1)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        let raw = Layout::array::<WorkspaceChange>(len)
            .map_err(|_| crate::context::host_allocation::HostAllocationError::Overflow)?
            .size();
        let claim = permit.reserve_exact_claim(raw)?;
        let changes = runmat_value::exact_capacity_vec_with(len, |index| {
            if index + 1 == len {
                appended.clone()
            } else {
                self.changes[index + usize::from(replaced.is_some_and(|old| index >= old))].clone()
            }
        })?;
        Ok(Self {
            changes,
            executor_suffix_start: self.executor_suffix_start.min(prefix_len),
            generation: self
                .generation
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?,
            _admitted_backing: Some(claim),
            prepared_impacts,
            prepared_impacts_complete: self.prepared_impacts_complete,
            has_prepared_mutation: true,
        })
    }

    fn executor_assignment_len(
        &self,
        name: &str,
    ) -> Result<usize, crate::context::host_allocation::HostAllocationError> {
        if self.changes[self.executor_suffix_start..]
            .iter()
            .any(|change| {
                matches!(
                    change.data(),
                    WorkspaceChangeData::Assign { name: previous, .. }
                        if previous.as_str() == name
                )
            })
        {
            Ok(self.changes.len())
        } else {
            self.changes
                .len()
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)
        }
    }

    pub(super) fn appended_change_len(
        &self,
    ) -> Result<usize, crate::context::host_allocation::HostAllocationError> {
        self.changes
            .len()
            .checked_add(1)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)
    }

    pub(super) fn executor_change_len(
        &self,
        name: &str,
    ) -> Result<usize, crate::context::host_allocation::HostAllocationError> {
        self.executor_assignment_len(name)
    }

    fn executor_assignment_replacement(&self, name: &str) -> Option<usize> {
        self.changes[self.executor_suffix_start..]
            .iter()
            .rposition(|change| {
                matches!(
                    change.data(),
                    WorkspaceChangeData::Assign { name: previous, .. }
                        if previous.as_str() == name
                )
            })
            .map(|offset| self.executor_suffix_start + offset)
    }

    fn prepare_legacy_mutation(&mut self) {
        self.prepared_impacts = PreparedWorkspaceImpacts::default();
        self.prepared_impacts_complete = false;
        let Some(claim) = self._admitted_backing.take() else {
            return;
        };
        let replacement = self.changes.clone();
        let admitted = std::mem::replace(&mut self.changes, replacement);
        drop(admitted);
        drop(claim);
    }

    pub(super) fn take_prepared_impacts(&mut self) -> PreparedWorkspaceImpacts {
        if self.prepared_impacts_complete {
            std::mem::take(&mut self.prepared_impacts)
        } else {
            self.prepared_impacts = PreparedWorkspaceImpacts::default();
            PreparedWorkspaceImpacts::default()
        }
    }

    pub(super) fn requires_history_reset(&self) -> bool {
        self.has_prepared_mutation && !self.prepared_impacts_complete
    }
}
