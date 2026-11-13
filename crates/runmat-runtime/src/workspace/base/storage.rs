//! Admitted immutable storage for canonical workspace generations.
//!
//! One-time imports sort exact-capacity scratch, then build a balanced
//! persistent binding tree with one exact claim per physical node. Published
//! generations share roots; executor hot writes still use the separate
//! coalescing overlay and never rebuild the full generation.

mod overlay;
mod binding_tree;
mod private_live;
use overlay::PreparedOverlayAssignment;
use overlay::WorkspaceOverlayLookup;
pub(super) use overlay::{WorkspaceNamespaceAction, WorkspaceOverlay};
pub(super) use private_live::{
    PreparedPrivateTransition, PrivateLiveGlobalNames, PrivateLiveLocalIter, PrivateLiveStorage,
};

use std::alloc::Layout;

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan, HostOutputPermit};
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit};

use super::prepared_write::PreparedWorkspaceWrite;
use super::values::{RootedWorkspaceEntries, RootedWorkspaceEntry};
use crate::workspace::retained::WorkspaceName;
use crate::workspace::retained::{WorkspaceValueLease, WorkspaceValueReadLease};
use crate::workspace::{GlobalBindingTarget, WorkspaceSlotReadLease};
use runmat_value::Value;

pub(super) use binding_tree::BindingTreeValue;
use binding_tree::{BindingTree, SortedBindingTreeError};

/// Allocation-free validation and exact node/sort-scratch quote for one
/// immutable local generation. The plan is bound to the entry count;
/// preparing it cannot discover an unsupported value after admission.
#[derive(Clone, Copy, Debug)]
pub(super) struct WorkspaceGenerationPlan {
    capacity: usize,
    retained_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct WorkspaceExecutorWritePlan {
    base: Option<WorkspaceGenerationPlan>,
    retained_bytes: u64,
}

/// The exact table (when legacy capture is needed) and overlay node for one
/// direct executor assignment. Both are allocated before visible mutation.
pub(super) struct PreparedStorageExecutorAssignment {
    base: Option<WorkspaceTableGeneration>,
    overlay: PreparedOverlayAssignment,
}

impl PreparedStorageExecutorAssignment {
    pub(super) fn name(&self) -> &WorkspaceName {
        self.overlay.name()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct WorkspaceNamespaceMutationPlan {
    base: Option<WorkspaceGenerationPlan>,
    retained_bytes: u64,
}

/// One exact, pre-effect migration of legacy and managed owners into a stable
/// generation. Existing Value and GC-root leases are shared, never retraced.
#[derive(Clone, Copy, Debug)]
pub(super) struct WorkspaceCapturePlan {
    base: WorkspaceGenerationPlan,
    retained_bytes: u64,
}

impl WorkspaceCapturePlan {
    pub(super) fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }
}

impl WorkspaceNamespaceMutationPlan {
    pub(super) fn captures_legacy_entries(self) -> bool {
        self.base.is_some()
    }

    pub(super) fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }

    pub(super) fn with_additional_retained_bytes(
        mut self,
        bytes: u64,
    ) -> Result<Self, HostAllocationError> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(self)
    }
}

impl WorkspaceExecutorWritePlan {
    pub(super) fn captures_legacy_entries(self) -> bool {
        self.base.is_some()
    }

    pub(super) fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }

    pub(super) fn with_additional_retained_bytes(
        mut self,
        bytes: u64,
    ) -> Result<Self, HostAllocationError> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(self)
    }
}

/// Exact owners retained by a projection and transferred to its canonical
/// owner at publication. The immutable base is prepared once; later executor
/// writes replace a unique matching overlay action or append across an actual
/// shared snapshot; neither path clones the full workspace.
#[derive(Clone, Debug, Default)]
pub(super) struct ManagedWorkspaceStorage {
    initialized: bool,
    view_current: bool,
    base: Option<WorkspaceTableGeneration>,
    committed: WorkspaceOverlay,
    pending: WorkspaceOverlay,
}

impl ManagedWorkspaceStorage {
    pub(super) fn is_prepared_live(&self) -> bool {
        self.initialized
            && self.view_current
            && self.committed.is_empty()
            && self.pending.is_empty()
    }

    pub(super) fn empty_live() -> Self {
        Self {
            initialized: true,
            view_current: true,
            base: None,
            committed: WorkspaceOverlay::default(),
            pending: WorkspaceOverlay::default(),
        }
    }

    /// Migrate only legacy namespace rows not superseded by this mutation.
    /// Local values were imported by the existing generation plan. Every
    /// surviving namespace row now shares the same live tree authority.
    pub(super) fn prepare_live_with_globals(
        &self,
        context: &crate::context::RuntimeContext,
        globals: &std::collections::HashMap<String, GlobalBindingTarget>,
        superseded: Option<&str>,
    ) -> Result<Self, HostAllocationError> {
        let mut live = self.prepare_live(Some(context))?;
        for (name, target) in globals {
            if superseded == Some(name.as_str()) || live.global(name).is_some() {
                continue;
            }
            let retained = claimed_name_bytes(name)?
                .checked_add(target.additional_retained_bytes()?)
                .ok_or(HostAllocationError::Overflow)?;
            let receipt = crate::context::host_allocation::HostAllocationReceipt::admit(
                Some(context),
                HostAllocationPlan::new(0, retained)?,
            )?;
            let (name, target) = match receipt {
                Some(receipt) => {
                    let permit = receipt.output_permit(context)?;
                    (
                        WorkspaceName::prepare(name, &permit)?,
                        target.prepare_retained(&permit)?,
                    )
                }
                None => (WorkspaceName::unbounded(name), target.clone()),
            };
            let tree = live
                .base
                .as_ref()
                .map(|base| base.0.clone())
                .unwrap_or_default();
            live.base = Some(WorkspaceTableGeneration(tree.prepare_set(
                Some(context),
                name,
                BindingTreeValue::Global(target),
            )?));
        }
        Ok(live)
    }

    /// Seal the current logical view without retaining overwritten overlay
    /// payloads. Only touched paths are copied; real captured trees keep their
    /// own roots. Call on an offstate candidate before installing it.
    pub(super) fn prepare_live(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<Self, HostAllocationError> {
        let tree = self
            .base
            .as_ref()
            .map(|base| base.0.clone())
            .unwrap_or_default();
        let tree = self.committed.prepare_tree(context, tree)?;
        let tree = self.pending.prepare_tree(context, tree)?;
        Ok(Self {
            initialized: self.initialized,
            view_current: self.view_current,
            base: Some(WorkspaceTableGeneration(tree)),
            committed: WorkspaceOverlay::default(),
            pending: WorkspaceOverlay::default(),
        })
    }

    /// Retain only the old published prefix while transferring this
    /// projection's unique pending suffix into a canonical publication.
    /// Cloning the suffix would make its tail non-unique and would turn
    /// attachment into an allocating copy-on-write operation.
    pub(super) fn retained_published_prefix(&self) -> Self {
        Self {
            initialized: self.initialized,
            view_current: self.view_current,
            base: self.base.clone(),
            committed: self.committed.clone(),
            pending: WorkspaceOverlay::default(),
        }
    }

    /// Commit an explicitly unbounded direct-owner action prepared off-state.
    /// Path-copy only overlay-touched bindings into the immutable tree. Keeping
    /// tombstones over an old base would make the current generation retain
    /// removed roots even when no actual snapshot still owns that generation.
    /// Preparation is fallible and off-state; installation still only swaps
    /// owners, and callers retire the displaced storage outside the Base borrow.
    pub(super) fn commit_unbounded_pending(&mut self) -> Result<(), HostAllocationError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let tree = self
            .base
            .as_ref()
            .map(|base| base.0.clone())
            .unwrap_or_default();
        let tree = self.committed.prepare_unbounded_tree(tree)?;
        let tree = self.pending.prepare_unbounded_tree(tree)?;
        self.base = Some(WorkspaceTableGeneration(tree));
        self.committed = WorkspaceOverlay::default();
        self.pending = WorkspaceOverlay::default();
        Ok(())
    }

    /// Capture the complete projected view under the explicitly unbounded
    /// policy. The returned representation is prepared off-state; callers
    /// install it only after publication participants accept the mutation.
    pub(super) fn prepare_unbounded_capture(
        &self,
        entries: &RootedWorkspaceEntries,
        globals: &std::collections::HashMap<String, GlobalBindingTarget>,
    ) -> Result<Self, HostAllocationError> {
        if self.is_canonical_without_legacy(entries, globals) {
            return Ok(self.clone());
        }
        let plan = self.plan_capture(entries, globals)?;
        self.prepare_capture(plan, entries, globals, None)
    }

    pub(super) fn stage_unbounded_owned_assignment(
        &mut self,
        name: WorkspaceName,
        entry: RootedWorkspaceEntry,
    ) {
        self.pending.stage_owned_unbounded(name, entry);
        self.initialized = true;
        self.view_current = true;
    }

    pub(super) fn stage_unbounded_namespace(
        &mut self,
        name: WorkspaceName,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
    ) {
        self.pending.stage_namespace_unbounded(name, action, target);
        self.initialized = true;
        self.view_current = true;
    }

    pub(super) fn stage_unbounded_clear(&mut self) {
        // Clear is a complete namespace barrier. Retaining either captured
        // base or an older overlay suffix would pin values no longer visible
        // in this projection; ordered pending effects keep their own roots.
        self.base = None;
        self.committed = WorkspaceOverlay::default();
        self.pending = WorkspaceOverlay::default();
        self.pending.stage_clear_unbounded();
        self.initialized = true;
        self.view_current = true;
    }
    pub(super) fn is_canonical_without_legacy(
        &self,
        entries: &RootedWorkspaceEntries,
        globals: &std::collections::HashMap<String, GlobalBindingTarget>,
    ) -> bool {
        self.initialized && self.view_current && entries.is_empty() && globals.is_empty()
    }

    pub(super) fn plan_capture(
        &self,
        entries: &RootedWorkspaceEntries,
        globals: &std::collections::HashMap<String, GlobalBindingTarget>,
    ) -> Result<WorkspaceCapturePlan, HostAllocationError> {
        let mut capacity = self.iter().count();
        for name in entries.keys() {
            if self.lookup(name.as_str()).is_none() {
                capacity = capacity
                    .checked_add(1)
                    .ok_or(HostAllocationError::Overflow)?;
            }
        }
        let mut retained_bytes = WorkspaceTableGeneration::retained_storage_bytes(capacity)?;
        for (name, _) in self.iter() {
            retained_bytes = retained_bytes
                .checked_add(claimed_name_bytes(name)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        for name in entries.keys() {
            if self.lookup(name.as_str()).is_none() {
                retained_bytes = retained_bytes
                    .checked_add(claimed_name_bytes(name.as_str())?)
                    .ok_or(HostAllocationError::Overflow)?;
            }
        }
        for name in self.global_names() {
            let target = self
                .global_target(name)
                .ok_or(HostAllocationError::StaleAuthority)?;
            let target_bytes = target.additional_retained_bytes()?;
            retained_bytes = retained_bytes
                .checked_add(WorkspaceOverlay::namespace_retained_bytes(name)?)
                .and_then(|bytes| bytes.checked_add(target_bytes))
                .ok_or(HostAllocationError::Overflow)?;
        }
        for (name, target) in globals {
            if self.global(name).is_none() {
                let target_bytes = target.additional_retained_bytes()?;
                retained_bytes = retained_bytes
                    .checked_add(WorkspaceOverlay::namespace_retained_bytes(name)?)
                    .and_then(|bytes| bytes.checked_add(target_bytes))
                    .ok_or(HostAllocationError::Overflow)?;
            }
        }
        Ok(WorkspaceCapturePlan {
            base: WorkspaceGenerationPlan {
                capacity,
                retained_bytes: 0,
            },
            retained_bytes,
        })
    }

    pub(super) fn prepare_capture(
        &self,
        plan: WorkspaceCapturePlan,
        entries: &RootedWorkspaceEntries,
        globals: &std::collections::HashMap<String, GlobalBindingTarget>,
        permit: Option<&HostOutputPermit>,
    ) -> Result<Self, HostAllocationError> {
        let base =
            WorkspaceTableGeneration::prepare_capture(plan.base, self, entries, permit.cloned())?;
        let mut committed = WorkspaceOverlay::default();
        for name in self.global_names() {
            let target = self
                .global_target(name)
                .ok_or(HostAllocationError::StaleAuthority)?;
            match permit {
                Some(permit) => committed.stage_namespace(
                    WorkspaceName::prepare(name, permit)?,
                    WorkspaceNamespaceAction::BindGlobal,
                    Some(target.prepare_retained(permit)?),
                    permit,
                )?,
                None => committed.stage_namespace_unbounded(
                    WorkspaceName::unbounded(name),
                    WorkspaceNamespaceAction::BindGlobal,
                    Some(target.clone()),
                ),
            }
        }
        for (name, target) in globals {
            if self.global(name).is_none() {
                match permit {
                    Some(permit) => committed.stage_namespace(
                        WorkspaceName::prepare(name, permit)?,
                        WorkspaceNamespaceAction::BindGlobal,
                        Some(target.prepare_retained(permit)?),
                        permit,
                    )?,
                    None => committed.stage_namespace_unbounded(
                        WorkspaceName::unbounded(name),
                        WorkspaceNamespaceAction::BindGlobal,
                        Some(target.clone()),
                    ),
                }
            }
        }
        Ok(Self {
            initialized: true,
            view_current: true,
            base: Some(base),
            committed,
            pending: WorkspaceOverlay::default(),
        })
    }

    pub(super) fn plan_namespace_mutation<'a>(
        &self,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        name: &str,
    ) -> Result<WorkspaceNamespaceMutationPlan, HostAllocationError> {
        let base = if self.initialized && self.view_current {
            None
        } else {
            let Some(plan) = WorkspaceGenerationPlan::for_values(existing)? else {
                return Err(HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ));
            };
            Some(plan)
        };
        let retained_bytes = base
            .map(WorkspaceGenerationPlan::retained_bytes)
            .unwrap_or(0)
            .checked_add(WorkspaceOverlay::namespace_retained_bytes(name)?)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(WorkspaceNamespaceMutationPlan {
            base,
            retained_bytes,
        })
    }

    pub(super) fn prepare_namespace_mutation<'a>(
        &mut self,
        plan: WorkspaceNamespaceMutationPlan,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        name: WorkspaceName,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
        permit: &HostOutputPermit,
    ) -> Result<(), HostAllocationError> {
        let base = match plan.base {
            Some(base) => Some(WorkspaceTableGeneration::prepare_values(
                base,
                existing,
                permit.clone(),
            )?),
            None => None,
        };
        if let Some(base) = base {
            let mut pending = WorkspaceOverlay::default();
            pending.stage_namespace(name.clone(), action, target, permit)?;
            self.base = Some(base);
            self.committed = WorkspaceOverlay::default();
            self.pending = pending;
            self.initialized = true;
            self.view_current = true;
        } else {
            self.pending
                .stage_namespace(name.clone(), action, target, permit)?;
        }
        Ok(())
    }

    pub(super) fn global(&self, name: &str) -> Option<bool> {
        self.pending
            .global(name)
            .or_else(|| self.committed.global(name))
            .or_else(|| {
                self.base
                    .as_ref()
                    .and_then(|base| base.0.get(name))
                    .map(|binding| matches!(binding, BindingTreeValue::Global(_)))
            })
    }

    pub(super) fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        self.pending
            .global_target(name)
            .or_else(|| self.committed.global_target(name))
            .or_else(|| self.base.as_ref().map(|base| base.global_target(name)))
            .flatten()
    }

    pub(super) fn plan_executor_write<'a>(
        &self,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        name: &str,
        value: &Value,
    ) -> Result<Option<WorkspaceExecutorWritePlan>, HostAllocationError> {
        let Some(overlay) = WorkspaceOverlay::retained_bytes(name, value)? else {
            return Ok(None);
        };
        self.plan_executor_write_with_overlay(existing, overlay)
    }

    pub(super) fn plan_executor_retained_write<'a>(
        &self,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        name: &str,
        source: &WorkspaceSlotReadLease,
    ) -> Result<Option<WorkspaceExecutorWritePlan>, HostAllocationError> {
        let Some(overlay) = WorkspaceOverlay::retained_generation_bytes(name, source)? else {
            return Ok(None);
        };
        self.plan_executor_write_with_overlay(existing, overlay)
    }

    fn plan_executor_write_with_overlay<'a>(
        &self,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        overlay: u64,
    ) -> Result<Option<WorkspaceExecutorWritePlan>, HostAllocationError> {
        let base = if self.initialized && self.view_current {
            None
        } else {
            let Some(plan) = WorkspaceGenerationPlan::for_values(existing)? else {
                return Ok(None);
            };
            Some(plan)
        };
        let retained_bytes = base
            .map(WorkspaceGenerationPlan::retained_bytes)
            .unwrap_or(0)
            .checked_add(overlay)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(Some(WorkspaceExecutorWritePlan {
            base,
            retained_bytes,
        }))
    }

    pub(super) fn prepare_executor_write<'a>(
        &mut self,
        plan: WorkspaceExecutorWritePlan,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        write: PreparedWorkspaceWrite,
        permit: &HostOutputPermit,
    ) -> Result<(), HostAllocationError> {
        let base = match plan.base {
            Some(base) => Some(WorkspaceTableGeneration::prepare_values(
                base,
                existing,
                permit.clone(),
            )?),
            None => None,
        };
        if let Some(base) = base {
            let mut pending = WorkspaceOverlay::default();
            pending.stage_prepared(write, permit)?;
            self.base = Some(base);
            self.committed = WorkspaceOverlay::default();
            self.pending = pending;
            self.initialized = true;
            self.view_current = true;
        } else {
            self.pending.stage_prepared(write, permit)?;
        }
        Ok(())
    }

    pub(super) fn prepare_executor_assignment<'a>(
        plan: WorkspaceExecutorWritePlan,
        existing: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        write: PreparedWorkspaceWrite,
        permit: &HostOutputPermit,
    ) -> Result<PreparedStorageExecutorAssignment, HostAllocationError> {
        let base = plan
            .base
            .map(|base| WorkspaceTableGeneration::prepare_values(base, existing, permit.clone()))
            .transpose()?;
        let overlay = WorkspaceOverlay::prepare_assignment(write, permit)?;
        Ok(PreparedStorageExecutorAssignment { base, overlay })
    }

    /// Install only preallocated owners. The returned old action is retired
    /// after the Base service borrow and observer callback have finished.
    pub(super) fn install_prepared_executor_assignment(
        &mut self,
        prepared: PreparedStorageExecutorAssignment,
    ) -> Option<WorkspaceOverlay> {
        if let Some(base) = prepared.base {
            self.base = Some(base);
            self.committed = WorkspaceOverlay::default();
            self.pending = WorkspaceOverlay::default();
            self.initialized = true;
            self.view_current = true;
        }
        self.pending.install_prepared_assignment(prepared.overlay)
    }

    pub(super) fn lookup(&self, name: &str) -> Option<&Value> {
        if !self.view_current {
            return None;
        }
        match self.pending.lookup(name) {
            WorkspaceOverlayLookup::PreparedValue(value) => Some(value.value()),
            WorkspaceOverlayLookup::OwnedValue(entry) => Some(entry.value()),
            WorkspaceOverlayLookup::Absent => None,
            WorkspaceOverlayLookup::Unmentioned => match self.committed.lookup(name) {
                WorkspaceOverlayLookup::PreparedValue(value) => Some(value.value()),
                WorkspaceOverlayLookup::OwnedValue(entry) => Some(entry.value()),
                WorkspaceOverlayLookup::Absent => None,
                WorkspaceOverlayLookup::Unmentioned => self
                    .base
                    .as_ref()
                    .and_then(|generation| generation.get(name))
                    .map(RootedWorkspaceEntry::value),
            },
        }
    }

    pub(super) fn read_lease(&self, name: &str) -> Option<WorkspaceValueReadLease> {
        if !self.view_current {
            return None;
        }
        match self.pending.lookup(name) {
            WorkspaceOverlayLookup::PreparedValue(value) => Some(WorkspaceValueReadLease::new(
                WorkspaceValueLease::from_retained(value.clone()),
            )),
            WorkspaceOverlayLookup::OwnedValue(entry) => Some(entry.read_lease()),
            WorkspaceOverlayLookup::Absent => None,
            WorkspaceOverlayLookup::Unmentioned => match self.committed.lookup(name) {
                WorkspaceOverlayLookup::PreparedValue(value) => Some(WorkspaceValueReadLease::new(
                    WorkspaceValueLease::from_retained(value.clone()),
                )),
                WorkspaceOverlayLookup::OwnedValue(entry) => Some(entry.read_lease()),
                WorkspaceOverlayLookup::Absent => None,
                WorkspaceOverlayLookup::Unmentioned => self
                    .base
                    .as_ref()
                    .and_then(|generation| generation.get(name))
                    .map(RootedWorkspaceEntry::read_lease),
            },
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Prepare affected canonical paths before revision installation. New
    /// nodes and legacy names use the supplied operation's exact admission;
    /// the subsequent visible install only swaps retained owner handles.
    pub(super) fn prepare_publication_over(
        &self,
        context: Option<&crate::context::RuntimeContext>,
        canonical: &Self,
        changes: &super::WorkspaceDelta,
    ) -> Result<Self, HostAllocationError> {
        let canonical = canonical.prepare_live(context)?;
        let mut tree = canonical
            .base
            .as_ref()
            .map(|base| base.0.clone())
            .unwrap_or_default();
        // The authentic delta names the affected paths. Resolve each name
        // from the final projection, preserving unrelated canonical writes.
        for change in changes.changes() {
            let Some(name) = change.entry_name() else {
                tree = BindingTree::default();
                continue;
            };
            let name = match name {
                super::values::WorkspaceEntryName::Prepared(name) => name.clone(),
                super::values::WorkspaceEntryName::Owned(name) => {
                    let retained = claimed_name_bytes(name)?;
                    match crate::context::host_allocation::HostAllocationReceipt::admit(
                        context,
                        HostAllocationPlan::new(0, retained)?,
                    )? {
                        Some(receipt) => WorkspaceName::prepare(
                            name,
                            &receipt.output_permit(
                                context.expect("managed name has an operation owner"),
                            )?,
                        )?,
                        None => WorkspaceName::unbounded(name),
                    }
                }
            };
            if let Some(target) = self.global_target(name.as_str()) {
                tree = tree.prepare_set(context, name, BindingTreeValue::Global(target.clone()))?;
            } else if let Some(entry) = self.entry(name.as_str()) {
                tree = tree.prepare_set(context, name, BindingTreeValue::Local(entry))?;
            } else {
                tree = tree.prepare_remove(context, name.as_str())?.unwrap_or(tree);
            }
        }
        Ok(Self {
            initialized: true,
            view_current: (!canonical.initialized || canonical.view_current) && self.view_current,
            base: Some(WorkspaceTableGeneration(tree)),
            committed: WorkspaceOverlay::default(),
            pending: WorkspaceOverlay::default(),
        })
    }

    pub(super) fn iter(&self) -> ManagedWorkspaceIter<'_> {
        ManagedWorkspaceIter {
            storage: self,
            previous: None,
        }
    }

    pub(super) fn global_names(&self) -> ManagedWorkspaceGlobalNames<'_> {
        ManagedWorkspaceGlobalNames {
            storage: self,
            previous: None,
        }
    }

    pub(super) fn entry(&self, name: &str) -> Option<RootedWorkspaceEntry> {
        if !self.view_current {
            return None;
        }
        match self.pending.lookup(name) {
            WorkspaceOverlayLookup::PreparedValue(value) => {
                Some(RootedWorkspaceEntry::from_retained(value.clone()))
            }
            WorkspaceOverlayLookup::OwnedValue(entry) => Some(entry.clone()),
            WorkspaceOverlayLookup::Absent => None,
            WorkspaceOverlayLookup::Unmentioned => match self.committed.lookup(name) {
                WorkspaceOverlayLookup::PreparedValue(value) => {
                    Some(RootedWorkspaceEntry::from_retained(value.clone()))
                }
                WorkspaceOverlayLookup::OwnedValue(entry) => Some(entry.clone()),
                WorkspaceOverlayLookup::Absent => None,
                WorkspaceOverlayLookup::Unmentioned => {
                    self.base.as_ref().and_then(|base| base.get(name)).cloned()
                }
            },
        }
    }
}

pub(super) struct ManagedWorkspaceGlobalNames<'a> {
    storage: &'a ManagedWorkspaceStorage,
    previous: Option<&'a str>,
}

impl<'a> Iterator for ManagedWorkspaceGlobalNames<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.storage.view_current {
            return None;
        }
        loop {
            let candidate = self
                .storage
                .base
                .as_ref()
                .and_then(|base| base.name_after(self.previous))
                .into_iter()
                .chain(self.storage.committed.names())
                .chain(self.storage.pending.names())
                .filter(|name| self.previous.is_none_or(|previous| *name > previous))
                .min()?;
            self.previous = Some(candidate);
            if self.storage.global(candidate) == Some(true) {
                return Some(candidate);
            }
        }
    }
}

pub(super) struct ManagedWorkspaceIter<'a> {
    storage: &'a ManagedWorkspaceStorage,
    previous: Option<&'a str>,
}

impl<'a> Iterator for ManagedWorkspaceIter<'a> {
    type Item = (&'a str, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.storage.view_current {
            return None;
        }
        loop {
            let candidate = self
                .storage
                .base
                .as_ref()
                .and_then(|base| base.name_after(self.previous))
                .into_iter()
                .chain(self.storage.committed.names())
                .chain(self.storage.pending.names())
                .filter(|name| self.previous.is_none_or(|previous| *name > previous))
                .min()?;
            self.previous = Some(candidate);
            if let Some(value) = self.storage.lookup(candidate) {
                return Some((candidate, value));
            }
        }
    }
}

fn claimed_name_bytes(name: &str) -> Result<u64, HostAllocationError> {
    let raw = WorkspaceName::requested_bytes(name)?;
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
        1,
    )
}

impl WorkspaceGenerationPlan {
    pub(super) fn for_values<'a>(
        entries: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
    ) -> Result<Option<Self>, HostAllocationError> {
        let capacity = entries.len();
        let mut retained_bytes = WorkspaceTableGeneration::retained_storage_bytes(capacity)?;
        for (_, name, value) in entries {
            let Some(write) = PreparedWorkspaceWrite::retained_bytes(name, value)? else {
                return Ok(None);
            };
            retained_bytes = retained_bytes
                .checked_add(write)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(Some(Self {
            capacity,
            retained_bytes,
        }))
    }

    pub(super) fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }
}

/// One immutable canonical local-value generation. The tree owns sorted
/// names and rooted entries together; there is no detached index or hash
/// table to keep in sync. Global namespace targets remain in the overlay.
#[derive(Clone, Debug)]
pub(super) struct WorkspaceTableGeneration(BindingTree);

/// Exact-capacity transient sorting scratch. The physical Vec backing drops
/// before its claim refunds, including when preparation fails halfway.
struct GenerationSortScratch {
    entries: Vec<(WorkspaceName, BindingTreeValue)>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl GenerationSortScratch {
    fn requested_bytes(len: usize) -> Result<usize, HostAllocationError> {
        Ok(Layout::array::<(WorkspaceName, BindingTreeValue)>(len)
            .map_err(|_| HostAllocationError::Overflow)?
            .size())
    }

    fn new(len: usize, permit: Option<&HostOutputPermit>) -> Result<Self, HostAllocationError> {
        if len == 0 {
            return Ok(Self {
                entries: Vec::new(),
                _claim: None,
            });
        }
        let layout = Layout::array::<(WorkspaceName, BindingTreeValue)>(len)
            .map_err(|_| HostAllocationError::Overflow)?;
        let claim = permit
            .map(|permit| permit.reserve_exact_claim(layout.size()))
            .transpose()?;
        let entries = exact_capacity_vec(len).map_err(HostAllocationError::Numeric)?;
        Ok(Self {
            entries,
            _claim: claim,
        })
    }

    fn push(
        &mut self,
        name: WorkspaceName,
        binding: BindingTreeValue,
    ) -> Result<(), HostAllocationError> {
        if self.entries.len() == self.entries.capacity() {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.entries.push((name, binding));
        Ok(())
    }

    fn finish(
        mut self,
        permit: Option<&HostOutputPermit>,
    ) -> Result<WorkspaceTableGeneration, HostAllocationError> {
        self.entries
            .sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let result = match permit {
            Some(permit) => BindingTree::prepare_sorted_with_permit(self.entries.drain(..), permit),
            None => BindingTree::prepare_sorted_exact(None, self.entries.drain(..)),
        };
        result
            .map(WorkspaceTableGeneration)
            .map_err(|error| match error {
                SortedBindingTreeError::Admission(error) => error,
                SortedBindingTreeError::NonAscendingNames
                | SortedBindingTreeError::MissingInput => HostAllocationError::StaleAuthority,
            })
    }
}

impl WorkspaceTableGeneration {
    fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        self.0.global_target(name)
    }

    pub(super) fn prepare_capture(
        plan: WorkspaceGenerationPlan,
        managed: &ManagedWorkspaceStorage,
        entries: &RootedWorkspaceEntries,
        permit: Option<HostOutputPermit>,
    ) -> Result<Self, HostAllocationError> {
        let mut scratch = GenerationSortScratch::new(plan.capacity, permit.as_ref())?;
        for (name, _) in managed.iter() {
            let entry = managed
                .entry(name)
                .ok_or(HostAllocationError::StaleAuthority)?;
            let name = match &permit {
                Some(permit) => WorkspaceName::prepare(name, permit)?,
                None => WorkspaceName::unbounded(name),
            };
            scratch.push(name, BindingTreeValue::Local(entry))?;
        }
        for (name, entry) in entries {
            if managed.lookup(name.as_str()).is_some() {
                continue;
            }
            // The entry clone shares its existing Value and GC-root owners.
            let name = match &permit {
                Some(permit) => WorkspaceName::prepare(name.as_str(), permit)?,
                None => WorkspaceName::unbounded(name.as_str()),
            };
            scratch.push(name, BindingTreeValue::Local(entry.clone()))?;
        }
        if scratch.entries.len() != plan.capacity {
            return Err(HostAllocationError::StaleAuthority);
        }
        scratch.finish(permit.as_ref())
    }

    fn retained_storage_bytes(capacity: usize) -> Result<u64, HostAllocationError> {
        let node_bytes = BindingTree::node_requested_bytes()
            .checked_mul(capacity)
            .ok_or(HostAllocationError::Overflow)?;
        let nodes = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(node_bytes).map_err(|_| HostAllocationError::Overflow)?,
            capacity,
        )?;
        if capacity == 0 {
            return Ok(nodes);
        }
        let scratch = GenerationSortScratch::requested_bytes(capacity)?;
        let scratch = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(scratch).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?;
        nodes
            .checked_add(scratch)
            .ok_or(HostAllocationError::Overflow)
    }

    pub(super) fn prepare_values<'a>(
        plan: WorkspaceGenerationPlan,
        entries: impl ExactSizeIterator<Item = (usize, &'a str, &'a Value)>,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        if entries.len() != plan.capacity {
            return Err(HostAllocationError::Overflow);
        }
        let mut scratch = GenerationSortScratch::new(plan.capacity, Some(&permit))?;
        for (slot, name, value) in entries {
            // Admission may call an external service before this projection
            // is borrowed for mutation. A reentrant change can invalidate
            // value eligibility even when the entry count stayed unchanged.
            let write = PreparedWorkspaceWrite::prepare(slot, name, value, &permit)?
                .ok_or(HostAllocationError::StaleAuthority)?;
            let (_, name, value) = write.into_parts();
            scratch.push(
                name,
                BindingTreeValue::Local(RootedWorkspaceEntry::from_retained(value)),
            )?;
        }
        if scratch.entries.len() != plan.capacity {
            return Err(HostAllocationError::StaleAuthority);
        }
        scratch.finish(Some(&permit))
    }

    pub(super) fn get(&self, name: &str) -> Option<&RootedWorkspaceEntry> {
        self.0.local(name)
    }

    fn name_after(&self, previous: Option<&str>) -> Option<&str> {
        self.0.name_after(previous).map(|(name, _)| name.as_str())
    }
}
