use std::collections::HashMap;
use std::rc::{Rc, Weak};

use runmat_value::Value;

use super::commit::PreparedDirectCommit;
use super::delta::{
    PreparedUnboundedDeltaTransition, PreparedWorkspaceDeltaMutation, PreparedWorkspaceDeltaView,
    WorkspaceChangeData, WorkspaceChangeStorage, WorkspaceMutationOrigin,
};
use super::owner::BaseWorkspaceState;
use super::storage::{
    ManagedWorkspaceGlobalNames, ManagedWorkspaceIter, ManagedWorkspaceStorage,
    PrivateLiveGlobalNames, PrivateLiveLocalIter, PrivateLiveStorage, WorkspaceExecutorWritePlan,
    WorkspaceNamespaceAction, WorkspaceNamespaceMutationPlan, WorkspaceOverlay,
};
use super::values::{RootedWorkspaceEntries, RootedWorkspaceEntry, WorkspaceEntryName};
use super::{
    BaseWorkspaceCommit, BaseWorkspaceError, BaseWorkspaceOwner, BaseWorkspaceReconciliation,
    WorkspaceChange, WorkspaceDelta, WorkspaceRevision,
};
use super::{PreparedWorkspaceWrite, WorkspaceName};
use crate::workspace::GlobalBindingTarget;
use crate::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};
use crate::workspace::{
    WorkspaceCompletionAction, WorkspaceCompletionSource, WorkspaceCompletionValue,
    WorkspaceSlotReadLease,
};

#[path = "projection/prepared_mutation.rs"]
mod prepared_mutation;
#[path = "projection/unbounded_mutation.rs"]
mod unbounded_mutation;
#[path = "projection/retirements.rs"]
mod retirements;
use prepared_mutation::PreparedProjectionAction;
pub(super) use prepared_mutation::{
    PreparedProjectionCompletion, PreparedProjectionMutation, PreparedProjectionPublicationView,
};
pub(in crate::workspace::base) use retirements::ProjectionRetirements;
pub(super) use unbounded_mutation::{
    PreparedUnboundedProjectionMutation, UnboundedProjectionAction,
};

#[cfg(test)]
#[path = "projection/executor_tests.rs"]
mod executor_tests;

#[derive(Debug)]
pub struct BaseWorkspaceProjection {
    owner: Weak<std::cell::RefCell<BaseWorkspaceState>>,
    revision: WorkspaceRevision,
    contents: ProjectionContents,
}

#[derive(Debug)]
enum ProjectionContents {
    Ordered(OrderedProjectionContents),
    Private(PrivateLiveStorage),
}

#[derive(Debug)]
struct OrderedProjectionContents {
    entries: RootedWorkspaceEntries,
    global_bindings: HashMap<String, GlobalBindingTarget>,
    pending: WorkspaceDelta,
    managed_storage: ManagedWorkspaceStorage,
}

pub(in crate::workspace::base) struct PreparedOrderedProjection {
    owner: Weak<std::cell::RefCell<BaseWorkspaceState>>,
    revision: WorkspaceRevision,
    contents: OrderedProjectionContents,
}

pub struct BaseWorkspaceProjectionIter<'a> {
    contents: ProjectionIterContents<'a>,
}

enum ProjectionIterContents<'a> {
    Ordered {
        managed: ManagedWorkspaceIter<'a>,
        legacy: std::collections::btree_map::Iter<'a, WorkspaceEntryName, RootedWorkspaceEntry>,
    },
    Private(PrivateLiveLocalIter<'a>),
}

impl<'a> Iterator for BaseWorkspaceProjectionIter<'a> {
    type Item = (&'a str, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.contents {
            ProjectionIterContents::Ordered { managed, legacy } => managed.next().or_else(|| {
                legacy
                    .next()
                    .map(|(name, entry)| (name.as_str(), entry.value()))
            }),
            ProjectionIterContents::Private(iter) => iter.next(),
        }
    }
}

pub struct BaseWorkspaceGlobalNames<'a> {
    contents: ProjectionGlobalNamesContents<'a>,
}

enum ProjectionGlobalNamesContents<'a> {
    Ordered {
        managed: ManagedWorkspaceGlobalNames<'a>,
        legacy: std::collections::hash_map::Keys<'a, String, GlobalBindingTarget>,
        storage: &'a ManagedWorkspaceStorage,
    },
    Private(PrivateLiveGlobalNames<'a>),
}

impl<'a> Iterator for BaseWorkspaceGlobalNames<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.contents {
            ProjectionGlobalNamesContents::Ordered {
                managed,
                legacy,
                storage,
            } => managed.next().or_else(|| {
                legacy
                    .find(|name| storage.global(name.as_str()).is_none())
                    .map(String::as_str)
            }),
            ProjectionGlobalNamesContents::Private(iter) => iter.next(),
        }
    }
}

pub(in crate::workspace::base) struct OrderedProjectionRef<'a> {
    revision: WorkspaceRevision,
    contents: &'a OrderedProjectionContents,
}

pub(in crate::workspace::base) struct OrderedProjectionMut<'a> {
    owner: &'a Weak<std::cell::RefCell<BaseWorkspaceState>>,
    revision: &'a mut WorkspaceRevision,
    contents: &'a mut OrderedProjectionContents,
}

impl<'a> OrderedProjectionRef<'a> {
    pub(in crate::workspace::base) fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub(in crate::workspace::base) fn pending(&self) -> &'a WorkspaceDelta {
        &self.contents.pending
    }

    pub(in crate::workspace::base) fn managed_storage(&self) -> &'a ManagedWorkspaceStorage {
        &self.contents.managed_storage
    }

    pub(in crate::workspace::base) fn entries(&self) -> &'a RootedWorkspaceEntries {
        &self.contents.entries
    }

    pub(in crate::workspace::base) fn global_bindings(
        &self,
    ) -> &'a HashMap<String, GlobalBindingTarget> {
        &self.contents.global_bindings
    }

    pub(in crate::workspace::base) fn lookup(&self, name: &str) -> Option<&'a Value> {
        if self.global_target(name).is_some() {
            return None;
        }
        self.contents.managed_storage.lookup(name).or_else(|| {
            self.contents
                .entries
                .get(name)
                .map(RootedWorkspaceEntry::value)
        })
    }

    pub(in crate::workspace::base) fn global_target(
        &self,
        name: &str,
    ) -> Option<&'a GlobalBindingTarget> {
        match self.contents.managed_storage.global(name) {
            Some(true) => self.contents.managed_storage.global_target(name),
            Some(false) => None,
            None => self.contents.global_bindings.get(name),
        }
    }
}

impl<'a> OrderedProjectionMut<'a> {
    pub(in crate::workspace::base) fn pending(&self) -> &WorkspaceDelta {
        &self.contents.pending
    }

    pub(in crate::workspace::base) fn pending_mut(&mut self) -> &mut WorkspaceDelta {
        &mut self.contents.pending
    }

    pub(in crate::workspace::base) fn managed_storage_mut(
        &mut self,
    ) -> &mut ManagedWorkspaceStorage {
        &mut self.contents.managed_storage
    }

    pub(in crate::workspace::base) fn entries_mut(&mut self) -> &mut RootedWorkspaceEntries {
        &mut self.contents.entries
    }

    pub(in crate::workspace::base) fn global_bindings_mut(
        &mut self,
    ) -> &mut HashMap<String, GlobalBindingTarget> {
        &mut self.contents.global_bindings
    }

    pub(in crate::workspace::base) fn install_completion(
        &mut self,
        prepared: PreparedProjectionCompletion,
    ) -> ProjectionRetirements {
        assert_eq!(*self.revision, prepared.expected_revision);
        assert_eq!(self.pending().generation(), prepared.expected_generation);
        let mut retired = ProjectionRetirements::default();
        if let Some(entries) = prepared.entries {
            retired.entries = Some(std::mem::replace(self.entries_mut(), entries));
        } else if prepared.captures_legacy {
            retired.entries = Some(std::mem::take(self.entries_mut()));
        }
        if let Some(global_bindings) = prepared.global_bindings {
            retired.globals = Some(std::mem::replace(
                self.global_bindings_mut(),
                global_bindings,
            ));
        }
        retired.storage = Some(std::mem::replace(
            self.managed_storage_mut(),
            prepared.managed_storage,
        ));
        retired.pending = Some(std::mem::replace(self.pending_mut(), prepared.pending));
        retired
    }

    pub(in crate::workspace::base) fn rollback_completion(
        &mut self,
        mut prior: ProjectionRetirements,
    ) -> ProjectionRetirements {
        let mut staged = ProjectionRetirements::default();
        if let Some(entries) = prior.entries.take() {
            staged.entries = Some(std::mem::replace(self.entries_mut(), entries));
        }
        if let Some(globals) = prior.globals.take() {
            staged.globals = Some(std::mem::replace(self.global_bindings_mut(), globals));
        }
        if let Some(storage) = prior.storage.take() {
            staged.storage = Some(std::mem::replace(self.managed_storage_mut(), storage));
        }
        if let Some(pending) = prior.pending.take() {
            staged.pending = Some(std::mem::replace(self.pending_mut(), pending));
        }
        staged
    }
}

impl BaseWorkspaceProjection {
    pub(super) fn new_private(owner: &BaseWorkspaceOwner) -> Result<Self, BaseWorkspaceError> {
        let state = owner
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        let storage = state
            .private_live
            .as_ref()
            .ok_or(BaseWorkspaceError::PrivateProjectionRequired)?
            .clone();
        Ok(Self {
            owner: Rc::downgrade(&owner.state),
            revision: state.revision,
            contents: ProjectionContents::Private(storage),
        })
    }

    pub(super) fn is_private(&self) -> bool {
        matches!(&self.contents, ProjectionContents::Private(_))
    }

    pub(super) fn set_private_revision(&mut self, revision: WorkspaceRevision) {
        debug_assert!(self.is_private());
        self.revision = revision;
    }

    pub(in crate::workspace::base) fn ordered_view(
        &self,
    ) -> Result<OrderedProjectionRef<'_>, BaseWorkspaceError> {
        match &self.contents {
            ProjectionContents::Ordered(contents) => Ok(OrderedProjectionRef {
                revision: self.revision,
                contents,
            }),
            ProjectionContents::Private(_) => Err(BaseWorkspaceError::PrivateProjectionRequired),
        }
    }

    fn ordered_allocation_view(
        &self,
    ) -> Result<OrderedProjectionRef<'_>, crate::context::host_allocation::HostAllocationError>
    {
        self.ordered_view().map_err(|_| {
            crate::context::host_allocation::HostAllocationError::Resource(
                crate::context::host_effect::HostEffectAllocationError::Unavailable,
            )
        })
    }

    pub(in crate::workspace::base) fn ordered_view_mut(
        &mut self,
    ) -> Result<OrderedProjectionMut<'_>, BaseWorkspaceError> {
        match &mut self.contents {
            ProjectionContents::Ordered(contents) => Ok(OrderedProjectionMut {
                owner: &self.owner,
                revision: &mut self.revision,
                contents,
            }),
            ProjectionContents::Private(_) => Err(BaseWorkspaceError::PrivateProjectionRequired),
        }
    }

    pub(in crate::workspace::base) fn private_storage(&self) -> Option<&PrivateLiveStorage> {
        match &self.contents {
            ProjectionContents::Private(storage) => Some(storage),
            ProjectionContents::Ordered(_) => None,
        }
    }

    pub(in crate::workspace::base) fn private_storage_mut(
        &mut self,
    ) -> Option<&mut PrivateLiveStorage> {
        match &mut self.contents {
            ProjectionContents::Private(storage) => Some(storage),
            ProjectionContents::Ordered(_) => None,
        }
    }

    /// Build the authentic completed candidate off-state. The managed
    /// generation and pending delta move from the prepared owner; only
    /// explicitly unbounded legacy tables can require a table clone here.
    pub(in crate::workspace::base) fn candidate_from_completion(
        &self,
        prepared: PreparedProjectionCompletion,
    ) -> Result<PreparedOrderedProjection, BaseWorkspaceError> {
        let ordered = self.ordered_view()?;
        assert_eq!(self.revision, prepared.expected_revision);
        assert_eq!(ordered.pending().generation(), prepared.expected_generation);
        assert!(prepared.captures_legacy || prepared.entries.is_none());
        Ok(PreparedOrderedProjection {
            owner: self.owner.clone(),
            revision: self.revision,
            contents: OrderedProjectionContents {
                entries: prepared.entries.unwrap_or_else(|| {
                    if prepared.captures_legacy {
                        RootedWorkspaceEntries::default()
                    } else {
                        ordered.entries().clone()
                    }
                }),
                global_bindings: prepared.global_bindings.unwrap_or_else(|| {
                    if prepared.captures_legacy {
                        HashMap::new()
                    } else {
                        ordered.global_bindings().clone()
                    }
                }),
                pending: prepared.pending,
                managed_storage: prepared.managed_storage,
            },
        })
    }
}

impl PreparedOrderedProjection {
    /// Prepare the canonical revision and effect independently of the live
    /// projection. The caller retains the original pending delta and old
    /// canonical backing until every composite participant has installed.
    pub(in crate::workspace::base) fn prepare_terminal_commit(
        &mut self,
        effect: WorkspaceDelta,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<PreparedDirectCommit, (BaseWorkspaceError, WorkspaceDelta)> {
        let Some(owner) = self.owner.upgrade() else {
            return Err((BaseWorkspaceError::Closed, effect));
        };
        let state = match owner.try_borrow() {
            Ok(state) => state,
            Err(_) => return Err((BaseWorkspaceError::BorrowConflict, effect)),
        };
        {
            if let Err(error) = state.preflight_commit(self.revision, &self.contents.pending) {
                return Err((error, effect));
            }
        }
        let prepared_live = self.revision == state.revision
            && self.contents.entries.is_empty()
            && self.contents.global_bindings.is_empty()
            && self.contents.managed_storage.is_prepared_live();
        let storage = match if prepared_live {
            Ok(self.contents.managed_storage.clone())
        } else {
            self.contents.managed_storage.prepare_publication_over(
                context,
                &state.managed_storage,
                &self.contents.pending,
            )
        } {
            Ok(storage) => storage,
            Err(error) => return Err((BaseWorkspaceError::HostAllocation(error), effect)),
        };
        let mut prepared = state
            .prepare_projection_commit(self.revision, &self.contents.pending, effect, storage)
            .expect("validated Base revision cannot change without a callback");
        drop(state);
        prepared.canonicalize_effect();
        Ok(prepared)
    }

    /// Adopt the already-installed canonical generation without dropping the
    /// authentic input suffix or any superseded roots under a workspace
    /// borrow. The returned owners retire after the composite observer phase.
    pub(in crate::workspace::base) fn finish_terminal_commit(
        mut self,
        storage: ManagedWorkspaceStorage,
        revision: WorkspaceRevision,
    ) -> (BaseWorkspaceProjection, ProjectionRetirements) {
        let mut retired = ProjectionRetirements {
            pending: Some(std::mem::take(&mut self.contents.pending)),
            storage: Some(std::mem::replace(
                &mut self.contents.managed_storage,
                storage,
            )),
            ..ProjectionRetirements::default()
        };
        if !self.contents.entries.is_empty() {
            retired.entries = Some(std::mem::take(&mut self.contents.entries));
        }
        if !self.contents.global_bindings.is_empty() {
            retired.globals = Some(std::mem::take(&mut self.contents.global_bindings));
        }
        self.revision = revision;
        (
            BaseWorkspaceProjection {
                owner: self.owner,
                revision: self.revision,
                contents: ProjectionContents::Ordered(self.contents),
            },
            retired,
        )
    }
}

impl BaseWorkspaceProjection {
    /// Retain the exact live managed view without publishing its pending
    /// overlay. Context-bound producers keep all names, values and namespace
    /// targets in this storage; legacy mirrors cannot be detached here under
    /// a finite policy because cloning their hash tables would allocate.
    pub(super) fn retained_live_snapshot(
        &self,
    ) -> Result<super::BaseWorkspaceSnapshot, BaseWorkspaceError> {
        self.ensure_effect_authority()?;
        let ordered = self.ordered_view()?;
        if !ordered.entries().is_empty() || !ordered.global_bindings().is_empty() {
            return Err(BaseWorkspaceError::HostAllocation(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            ));
        }
        Ok(super::BaseWorkspaceSnapshot::from_managed_projection(
            self.revision,
            ordered.managed_storage().clone(),
        ))
    }

    pub(super) fn retained_live_snapshot_unbounded(
        &self,
    ) -> Result<super::BaseWorkspaceSnapshot, BaseWorkspaceError> {
        self.ensure_effect_authority()?;
        let ordered = self.ordered_view()?;
        let storage = ordered
            .managed_storage()
            .prepare_unbounded_capture(ordered.entries(), ordered.global_bindings())
            .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(super::BaseWorkspaceSnapshot::from_managed_projection(
            self.revision,
            storage,
        ))
    }
    pub(super) fn completion_retained_bytes(
        &self,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let ordered = self.ordered_view().map_err(|_| {
            crate::context::host_allocation::HostAllocationError::Resource(
                crate::context::host_effect::HostEffectAllocationError::Unavailable,
            )
        })?;
        self.validate_completion_source(source)?;
        let mut retained = 0_u64;
        for index in 0..source.len() {
            let action = source.action(index).ok_or(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            )?;
            let action_retained = if index == 0 {
                match &action {
                    WorkspaceCompletionAction::AssignLocal { name, value } => {
                        let existing = || {
                            ordered
                                .entries()
                                .iter()
                                .enumerate()
                                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value()))
                        };
                        match value {
                            WorkspaceCompletionValue::Borrowed(value) => ordered
                                .managed_storage()
                                .plan_executor_write(existing(), name, value)?,
                            WorkspaceCompletionValue::Retained(source) => ordered
                                .managed_storage()
                                .plan_executor_retained_write(existing(), name, source)?,
                        }
                        .ok_or(
                            crate::context::host_allocation::HostAllocationError::Resource(
                                crate::context::host_effect::HostEffectAllocationError::Unavailable,
                            ),
                        )?
                        .retained_bytes()
                    }
                    WorkspaceCompletionAction::BindGlobal { name, .. }
                    | WorkspaceCompletionAction::UnbindGlobal { name }
                    | WorkspaceCompletionAction::Remove { name } => {
                        ordered
                            .managed_storage()
                            .plan_namespace_mutation(
                                ordered.entries().iter().enumerate().map(
                                    |(slot, (name, entry))| (slot, name.as_str(), entry.value()),
                                ),
                                name,
                            )?
                            .retained_bytes()
                    }
                }
            } else {
                match &action {
                    WorkspaceCompletionAction::AssignLocal { name, value } => {
                        let bytes = match value {
                            WorkspaceCompletionValue::Borrowed(value) => {
                                WorkspaceOverlay::retained_bytes(name, value)?
                            }
                            WorkspaceCompletionValue::Retained(source) => {
                                WorkspaceOverlay::retained_generation_bytes(name, source)?
                            }
                        };
                        bytes.ok_or(
                            crate::context::host_allocation::HostAllocationError::Resource(
                                crate::context::host_effect::HostEffectAllocationError::Unavailable,
                            ),
                        )?
                    }
                    WorkspaceCompletionAction::BindGlobal { name, .. }
                    | WorkspaceCompletionAction::UnbindGlobal { name }
                    | WorkspaceCompletionAction::Remove { name } => {
                        WorkspaceOverlay::namespace_retained_bytes(name)?
                    }
                }
            };
            let action_retained = match &action {
                WorkspaceCompletionAction::BindGlobal { target, .. } => action_retained
                    .checked_add(target.additional_retained_bytes()?)
                    .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?,
                _ => action_retained,
            };
            retained = retained
                .checked_add(action_retained)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        }
        let delta = ordered.pending().completion_retained_bytes(
            source.len(),
            (0..source.len()).map(|index| {
                source
                    .action(index)
                    .expect("validated completion source is stable")
                    .name()
            }),
        )?;
        let effect = WorkspaceDelta::independent_effect_retained_bytes(
            ordered
                .pending()
                .changes()
                .len()
                .checked_add(source.len())
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?,
        )?;
        retained
            .checked_add(delta)
            .and_then(|retained| retained.checked_add(effect))
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)
    }

    pub(super) fn prepare_completion(
        &self,
        context: &crate::context::RuntimeContext,
        source: &dyn WorkspaceCompletionSource,
        permit: &crate::context::host_allocation::HostOutputPermit,
    ) -> Result<PreparedProjectionCompletion, BaseWorkspaceError> {
        self.validate_completion_source(source)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let ordered = self.ordered_view()?;
        let source_start = ordered.pending().changes().len();
        let mut managed_storage = ordered.managed_storage().clone();
        let mut delta = ordered
            .pending()
            .prepare_completion(source.len(), permit)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let mut captures_legacy = false;
        for index in 0..source.len() {
            let action = source.action(index).ok_or({
                BaseWorkspaceError::HostAllocation(
                    crate::context::host_allocation::HostAllocationError::Resource(
                        crate::context::host_effect::HostEffectAllocationError::Unavailable,
                    ),
                )
            })?;
            let change =
                match action {
                    WorkspaceCompletionAction::AssignLocal { name, value } => {
                        let existing = || {
                            ordered
                                .entries()
                                .iter()
                                .enumerate()
                                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value()))
                        };
                        let plan = match &value {
                            WorkspaceCompletionValue::Borrowed(value) => {
                                managed_storage.plan_executor_write(existing(), name, value)
                            }
                            WorkspaceCompletionValue::Retained(source) => managed_storage
                                .plan_executor_retained_write(existing(), name, source),
                        }
                        .map_err(BaseWorkspaceError::HostAllocation)?
                        .ok_or({
                            BaseWorkspaceError::HostAllocation(
                            crate::context::host_allocation::HostAllocationError::Resource(
                                crate::context::host_effect::HostEffectAllocationError::Unavailable,
                            ),
                        )
                        })?;
                        captures_legacy |= plan.captures_legacy_entries();
                        let write = match value {
                            WorkspaceCompletionValue::Borrowed(value) => {
                                PreparedWorkspaceWrite::prepare(0, name, value, permit)
                            }
                            WorkspaceCompletionValue::Retained(source) => {
                                PreparedWorkspaceWrite::prepare_retained(0, name, source, permit)
                            }
                        }
                        .map_err(crate::context::host_allocation::HostAllocationError::from)
                        .map_err(BaseWorkspaceError::HostAllocation)?
                        .ok_or({
                            BaseWorkspaceError::HostAllocation(
                            crate::context::host_allocation::HostAllocationError::Resource(
                                crate::context::host_effect::HostEffectAllocationError::Unavailable,
                            ),
                        )
                        })?;
                        let entry_name = WorkspaceEntryName::Prepared(write.name().clone());
                        let entry = RootedWorkspaceEntry::from_prepared(&write);
                        managed_storage
                            .prepare_executor_write(
                                plan,
                                ordered.entries().iter().enumerate().map(
                                    |(slot, (name, entry))| (slot, name.as_str(), entry.value()),
                                ),
                                write,
                                permit,
                            )
                            .map_err(BaseWorkspaceError::HostAllocation)?;
                        WorkspaceChange::assign_entry(entry_name, entry)
                    }
                    WorkspaceCompletionAction::BindGlobal { name, .. }
                    | WorkspaceCompletionAction::UnbindGlobal { name }
                    | WorkspaceCompletionAction::Remove { name } => {
                        let plan = managed_storage
                            .plan_namespace_mutation(
                                ordered.entries().iter().enumerate().map(
                                    |(slot, (name, entry))| (slot, name.as_str(), entry.value()),
                                ),
                                name,
                            )
                            .map_err(BaseWorkspaceError::HostAllocation)?;
                        captures_legacy |= plan.captures_legacy_entries();
                        let prepared_name = WorkspaceName::prepare(name, permit)
                            .map_err(crate::context::host_allocation::HostAllocationError::from)
                            .map_err(BaseWorkspaceError::HostAllocation)?;
                        let entry_name = WorkspaceEntryName::Prepared(prepared_name.clone());
                        let target = match action {
                            WorkspaceCompletionAction::BindGlobal { target, .. } => Some(
                                target
                                    .prepare_retained(permit)
                                    .map_err(
                                        crate::context::host_allocation::HostAllocationError::from,
                                    )
                                    .map_err(BaseWorkspaceError::HostAllocation)?,
                            ),
                            _ => None,
                        };
                        let namespace_action = match action {
                            WorkspaceCompletionAction::BindGlobal { .. } => {
                                WorkspaceNamespaceAction::BindGlobal
                            }
                            WorkspaceCompletionAction::UnbindGlobal { .. } => {
                                WorkspaceNamespaceAction::UnbindGlobal
                            }
                            WorkspaceCompletionAction::Remove { .. } => {
                                WorkspaceNamespaceAction::Remove
                            }
                            WorkspaceCompletionAction::AssignLocal { .. } => unreachable!(),
                        };
                        managed_storage
                            .prepare_namespace_mutation(
                                plan,
                                ordered.entries().iter().enumerate().map(
                                    |(slot, (name, entry))| (slot, name.as_str(), entry.value()),
                                ),
                                prepared_name,
                                namespace_action,
                                target.clone(),
                                permit,
                            )
                            .map_err(BaseWorkspaceError::HostAllocation)?;
                        match action {
                            WorkspaceCompletionAction::BindGlobal { .. } => {
                                WorkspaceChange::bind_global_prepared(
                                    entry_name,
                                    target.expect("prepared global target"),
                                )
                            }
                            WorkspaceCompletionAction::UnbindGlobal { .. } => {
                                WorkspaceChange::unbind_global_prepared(entry_name)
                            }
                            WorkspaceCompletionAction::Remove { .. } => {
                                WorkspaceChange::remove_prepared(entry_name)
                            }
                            WorkspaceCompletionAction::AssignLocal { .. } => unreachable!(),
                        }
                    }
                };
            delta
                .push(change.staged_in_managed_overlay(), permit)
                .map_err(BaseWorkspaceError::HostAllocation)?;
        }
        let pending = delta.finish().map_err(BaseWorkspaceError::HostAllocation)?;
        if !captures_legacy
            && pending.changes()[source_start..]
                .iter()
                .filter_map(WorkspaceChange::name)
                .any(|name| ordered.entries().contains_key(name))
        {
            return Err(BaseWorkspaceError::HostAllocation(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            ));
        }
        Ok(PreparedProjectionCompletion {
            entries: None,
            global_bindings: None,
            managed_storage: managed_storage
                .prepare_live(Some(context))
                .map_err(BaseWorkspaceError::HostAllocation)?,
            pending,
            captures_legacy,
            source_start,
            expected_revision: self.revision,
            expected_generation: ordered.pending().generation(),
        })
    }

    /// Prepare one complete replacement under the explicit unbounded host
    /// policy. Rooting and ordinary collection allocation happen against
    /// detached state before the live projection changes.
    pub(super) fn prepare_completion_unbounded(
        &self,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<PreparedProjectionCompletion, BaseWorkspaceError> {
        self.validate_completion_source(source)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let ordered = self.ordered_view()?;
        let source_start = ordered.pending().changes().len();
        let mut managed_storage = ordered
            .managed_storage()
            .prepare_unbounded_capture(ordered.entries(), ordered.global_bindings())
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let mut pending = ordered.pending().clone_unbounded();
        for index in 0..source.len() {
            let action = source.action(index).ok_or({
                BaseWorkspaceError::HostAllocation(
                    crate::context::host_allocation::HostAllocationError::Resource(
                        crate::context::host_effect::HostEffectAllocationError::Unavailable,
                    ),
                )
            })?;
            let change = match action {
                WorkspaceCompletionAction::AssignLocal { name, value } => {
                    let entry = match value {
                        WorkspaceCompletionValue::Borrowed(value) => {
                            RootedWorkspaceEntry::new(value.clone())?
                        }
                        WorkspaceCompletionValue::Retained(source) => {
                            RootedWorkspaceEntry::from_read_lease(source)
                        }
                    };
                    let retained_name = WorkspaceName::unbounded(name);
                    managed_storage.stage_unbounded_namespace(
                        retained_name.clone(),
                        WorkspaceNamespaceAction::UnbindGlobal,
                        None,
                    );
                    managed_storage
                        .stage_unbounded_owned_assignment(retained_name.clone(), entry.clone());
                    WorkspaceChange::assign_entry(
                        WorkspaceEntryName::Prepared(retained_name),
                        entry,
                    )
                    .staged_in_managed_overlay()
                }
                WorkspaceCompletionAction::BindGlobal { name, target } => {
                    let retained_name = WorkspaceName::unbounded(name);
                    managed_storage.stage_unbounded_namespace(
                        retained_name.clone(),
                        WorkspaceNamespaceAction::BindGlobal,
                        Some(target.clone()),
                    );
                    WorkspaceChange::bind_global_prepared(
                        WorkspaceEntryName::Prepared(retained_name),
                        target.clone(),
                    )
                    .staged_in_managed_overlay()
                }
                WorkspaceCompletionAction::UnbindGlobal { name } => {
                    let retained_name = WorkspaceName::unbounded(name);
                    managed_storage.stage_unbounded_namespace(
                        retained_name.clone(),
                        WorkspaceNamespaceAction::UnbindGlobal,
                        None,
                    );
                    WorkspaceChange::unbind_global_prepared(WorkspaceEntryName::Prepared(
                        retained_name,
                    ))
                    .staged_in_managed_overlay()
                }
                WorkspaceCompletionAction::Remove { name } => {
                    if managed_storage.global(name) != Some(true) {
                        let retained_name = WorkspaceName::unbounded(name);
                        managed_storage.stage_unbounded_namespace(
                            retained_name.clone(),
                            WorkspaceNamespaceAction::Remove,
                            None,
                        );
                        WorkspaceChange::remove_prepared(WorkspaceEntryName::Prepared(
                            retained_name,
                        ))
                        .staged_in_managed_overlay()
                    } else {
                        WorkspaceChange::remove_prepared(WorkspaceEntryName::Owned(name.to_owned()))
                    }
                }
            };
            let transition = pending
                .prepare_unbounded_transition(change, None, false, WorkspaceMutationOrigin::Ordered)
                .map_err(BaseWorkspaceError::HostAllocation)?;
            pending
                .reserve_unbounded_transition(&transition)
                .map_err(BaseWorkspaceError::HostAllocation)?;
            drop(pending.install_unbounded_transition(transition));
        }
        Ok(PreparedProjectionCompletion {
            entries: None,
            global_bindings: None,
            managed_storage,
            pending,
            captures_legacy: true,
            source_start,
            expected_revision: self.revision,
            expected_generation: ordered.pending().generation(),
        })
    }

    pub(in crate::workspace::base) fn validate_completion_source(
        &self,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<(), crate::context::host_allocation::HostAllocationError> {
        let mut previous = None;
        for index in 0..source.len() {
            let action = source.action(index).ok_or(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            )?;
            let name = action.name();
            if name.is_empty() || previous.is_some_and(|previous| previous >= name) {
                return Err(
                    crate::context::host_allocation::HostAllocationError::Resource(
                        crate::context::host_effect::HostEffectAllocationError::Unavailable,
                    ),
                );
            }
            if let WorkspaceCompletionAction::AssignLocal { value, .. } = action {
                crate::execution::validate_storable_value(value.value()).map_err(|_| {
                    crate::context::host_allocation::HostAllocationError::Resource(
                        crate::context::host_effect::HostEffectAllocationError::Unavailable,
                    )
                })?;
            }
            previous = Some(name);
        }
        Ok(())
    }

    /// Retire a selection at the owner revision, with no failure point after
    /// local mutation. The retired values are rooted before publication.
    pub fn detach_selected(
        &mut self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<(DetachedWorkspaceValues, BaseWorkspaceCommit), BaseWorkspaceError> {
        self.capture_canonical_prefix(None)?;
        let detached = self.logical_detach_values_unbounded(selection)?;
        let delta = self.prepare_detach_selected(selection)?;
        let completion = self.prepare_detachment_completion(delta)?;
        let commit = self.commit_detached(completion)?;
        Ok((detached, commit))
    }

    pub(super) fn logical_detach_values_unbounded(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, BaseWorkspaceError> {
        let mut values: Vec<_> = self
            .iter()
            .filter(|(name, _)| self.global_target(name).is_none())
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect();
        for name in self.global_names() {
            let Some(target) = self.global_target(name) else {
                continue;
            };
            let value = match target {
                GlobalBindingTarget::Named(target) => {
                    crate::workspace::session::global_value(target.as_str())
                }
                GlobalBindingTarget::LegacySlot(slot) => {
                    crate::workspace::session::global_slot_value(*slot)
                }
            };
            if let Some(value) = value {
                values.push((name.to_owned(), value));
            }
        }
        DetachedWorkspaceValues::new(selection.select(values))
    }

    pub(super) fn prepare_detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<WorkspaceDelta, BaseWorkspaceError> {
        if self.has_pending_changes() {
            return Err(BaseWorkspaceError::PendingChanges);
        }
        let mut delta = WorkspaceDelta::new();
        match selection {
            WorkspaceDetachSelection::All => delta.clear(),
            WorkspaceDetachSelection::Names(names) => {
                for name in names {
                    if self.is_global(name) {
                        delta.unbind_global(name.clone());
                    } else {
                        delta.remove(name.clone());
                    }
                }
            }
            WorkspaceDetachSelection::Except(excluded) => {
                let excluded: std::collections::HashSet<_> =
                    excluded.iter().map(String::as_str).collect();
                for (name, _) in self.iter() {
                    if self.global_target(name).is_none() && !excluded.contains(name) {
                        delta.remove(name.to_owned());
                    }
                }
                for name in self.global_names() {
                    if !excluded.contains(name) {
                        delta.unbind_global(name.to_owned());
                    }
                }
            }
        }
        Ok(delta)
    }

    /// Stage an already rooted unbounded detachment as one ordinary pending
    /// completion. The detached payload is owned separately by the caller;
    /// only name/namespace effects enter this projection candidate.
    pub(super) fn prepare_detachment_completion(
        &self,
        mut delta: WorkspaceDelta,
    ) -> Result<PreparedProjectionCompletion, BaseWorkspaceError> {
        if self.has_pending_changes() {
            return Err(BaseWorkspaceError::PendingChanges);
        }
        delta.normalize_detachment_names();
        let source_start = usize::from(
            delta
                .changes()
                .first()
                .is_some_and(|change| change.kind() == super::WorkspaceChangeKind::Clear),
        );
        let ordered = self.ordered_view()?;
        let mut managed_storage = ordered
            .managed_storage()
            .prepare_unbounded_capture(ordered.entries(), ordered.global_bindings())
            .map_err(BaseWorkspaceError::HostAllocation)?;
        for change in delta.changes() {
            match change.data() {
                WorkspaceChangeData::Remove { name } => managed_storage.stage_unbounded_namespace(
                    WorkspaceName::unbounded(name.as_str()),
                    WorkspaceNamespaceAction::Remove,
                    None,
                ),
                WorkspaceChangeData::UnbindGlobal { name } => {
                    managed_storage.stage_unbounded_namespace(
                        WorkspaceName::unbounded(name.as_str()),
                        WorkspaceNamespaceAction::UnbindGlobal,
                        None,
                    );
                }
                WorkspaceChangeData::Clear => managed_storage.stage_unbounded_clear(),
                _ => unreachable!("detachment contains only retirement actions"),
            }
        }
        delta.mark_detachment_overlay_backed();
        Ok(PreparedProjectionCompletion {
            entries: Some(RootedWorkspaceEntries::new()),
            global_bindings: Some(HashMap::new()),
            managed_storage,
            pending: delta,
            captures_legacy: true,
            source_start,
            expected_revision: self.revision,
            expected_generation: ordered.pending().generation(),
        })
    }

    pub(super) fn commit_detached(
        &mut self,
        completion: PreparedProjectionCompletion,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let effect = completion.prepare_effect_unbounded();
        let mut ordered = self.ordered_view_mut()?;
        let retired = ordered.install_completion(completion);
        match ordered.publish_with_effect(effect, None) {
            Ok((commit, published_retired)) => {
                drop(published_retired);
                drop(retired);
                Ok(commit)
            }
            Err((error, _effect)) => {
                let staged = ordered.rollback_completion(retired);
                drop(staged);
                Err(error)
            }
        }
    }

    pub(super) fn new(owner: &BaseWorkspaceOwner) -> Result<Self, BaseWorkspaceError> {
        let state = owner
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(Self {
            owner: Rc::downgrade(&owner.state),
            revision: state.revision,
            contents: ProjectionContents::Ordered(OrderedProjectionContents {
                entries: state.entries.clone(),
                global_bindings: state.global_bindings.clone(),
                pending: WorkspaceDelta::new(),
                managed_storage: state.managed_storage.clone(),
            }),
        })
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn len(&self) -> usize {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => {
                ordered.managed_storage.iter().count() + ordered.entries.len()
            }
            ProjectionContents::Private(storage) => storage.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => {
                ordered.managed_storage.iter().next().is_none() && ordered.entries.is_empty()
            }
            ProjectionContents::Private(storage) => storage.is_empty(),
        }
    }

    /// Iterate the projection's retained local view. The view remains valid
    /// after its owner closes so diagnostics and unwind code can inspect
    /// already-admitted values; it does not confer executable authority.
    pub fn iter(&self) -> BaseWorkspaceProjectionIter<'_> {
        let contents = match &self.contents {
            ProjectionContents::Ordered(ordered) => ProjectionIterContents::Ordered {
                managed: ordered.managed_storage.iter(),
                legacy: ordered.entries.iter(),
            },
            ProjectionContents::Private(storage) => {
                ProjectionIterContents::Private(storage.iter_local())
            }
        };
        BaseWorkspaceProjectionIter { contents }
    }

    /// Read the projection's retained local view. Call
    /// [`Self::ensure_effect_authority`] before admitting new execution or an
    /// external effect against this projection.
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => {
                let view = OrderedProjectionRef {
                    revision: self.revision,
                    contents: ordered,
                };
                view.lookup(name)
            }
            ProjectionContents::Private(storage) => storage.value(name),
        }
    }

    pub fn is_global(&self, name: &str) -> bool {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => ordered
                .managed_storage
                .global(name)
                .unwrap_or_else(|| ordered.global_bindings.contains_key(name)),
            ProjectionContents::Private(storage) => storage.global_target(name).is_some(),
        }
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => match ordered.managed_storage.global(name) {
                Some(true) => ordered.managed_storage.global_target(name),
                Some(false) => None,
                None => ordered.global_bindings.get(name),
            },
            ProjectionContents::Private(storage) => storage.global_target(name),
        }
    }

    pub fn global_names(&self) -> BaseWorkspaceGlobalNames<'_> {
        let contents = match &self.contents {
            ProjectionContents::Ordered(ordered) => ProjectionGlobalNamesContents::Ordered {
                managed: ordered.managed_storage.global_names(),
                legacy: ordered.global_bindings.keys(),
                storage: &ordered.managed_storage,
            },
            ProjectionContents::Private(storage) => {
                ProjectionGlobalNamesContents::Private(storage.global_names())
            }
        };
        BaseWorkspaceGlobalNames { contents }
    }

    /// Verify that the canonical owner which admitted this projection is
    /// still live. Local reads remain available after closure, but publication,
    /// reconciliation, and new executor/effect admission must pass this check.
    pub fn ensure_effect_authority(&self) -> Result<(), BaseWorkspaceError> {
        let owner = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        let state = owner
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()
    }

    /// Capture the exact canonical prefix before a prospective mutation. A
    /// physical capture changes no semantic revision, but prevents final
    /// publication from discovering a legacy map allocation after effects.
    pub(super) fn capture_canonical_prefix(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<(), BaseWorkspaceError> {
        let owner = self.canonical_owner()?;
        match context {
            Some(context) => {
                owner.capture_stable_with_context(context)?;
            }
            None => {
                owner.capture_stable_unbounded()?;
            }
        }
        Ok(())
    }

    pub(super) fn canonical_owner(&self) -> Result<BaseWorkspaceOwner, BaseWorkspaceError> {
        let state = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        Ok(BaseWorkspaceOwner { state })
    }

    pub fn has_pending_changes(&self) -> bool {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => {
                !ordered.pending.is_empty() || ordered.managed_storage.has_pending()
            }
            // Private transitions atomically update the canonical pruned tree
            // and the live projection. Its hot entry is authoritative state,
            // not an unpublished ordered suffix.
            ProjectionContents::Private(_) => false,
        }
    }

    pub fn assign(
        &mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<(), BaseWorkspaceError> {
        let name = name.into();
        drop(self.assign_unbounded_retired(&name, value, None, false, false)?);
        Ok(())
    }

    pub(super) fn assign_unbounded_retired(
        &mut self,
        name: &str,
        value: Value,
        slot: Option<usize>,
        local: bool,
        coalesce: bool,
    ) -> Result<ProjectionRetirements, BaseWorkspaceError> {
        let entry = RootedWorkspaceEntry::new(value)?;
        self.capture_canonical_prefix(None)?;
        let prepared = self
            .prepare_unbounded_assignment_entry_with_order(name, &entry, slot, local, coalesce)?;
        self.reserve_unbounded_mutation_capacity(&prepared)?;
        Ok(self
            .ordered_view_mut()?
            .install_unbounded_mutation(prepared))
    }

    pub(super) fn prepare_assignment(
        &self,
        context: &crate::context::RuntimeContext,
        plan: WorkspaceExecutorWritePlan,
        write: &PreparedWorkspaceWrite,
        permit: &crate::context::host_allocation::HostOutputPermit,
        executor_origin: bool,
    ) -> Result<PreparedProjectionMutation, BaseWorkspaceError> {
        let retained_name = write.name().clone();
        let slot = write.slot();
        let entry = RootedWorkspaceEntry::from_prepared(write);
        let entry_name = WorkspaceEntryName::Prepared(retained_name.clone());
        let pending_mutation = self
            .ordered_view()?
            .contents
            .pending
            .prepare_executor_assignment(entry_name, entry, executor_origin.then_some(slot), permit)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let captures_legacy = plan.captures_legacy_entries();
        if !captures_legacy
            && self
                .ordered_view()?
                .contents
                .entries
                .contains_key(retained_name.as_str())
        {
            return Err(BaseWorkspaceError::HostAllocation(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            ));
        }
        let storage = ManagedWorkspaceStorage::prepare_executor_assignment(
            plan,
            self.ordered_view()?
                .contents
                .entries
                .iter()
                .enumerate()
                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value())),
            write.clone(),
            permit,
        )
        .map_err(BaseWorkspaceError::HostAllocation)?;
        let name = storage.name().clone();
        let mut live = self.ordered_view()?.contents.managed_storage.clone();
        drop(live.install_prepared_executor_assignment(storage));
        let live = live
            .prepare_live_with_globals(
                context,
                &self.ordered_view()?.contents.global_bindings,
                Some(name.as_str()),
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(PreparedProjectionMutation {
            live,
            pending: pending_mutation,
            action: PreparedProjectionAction::Named { name },
            captures_legacy,
            expected_revision: self.revision,
            expected_generation: self.ordered_view()?.contents.pending.generation(),
        })
    }

    pub(super) fn install_prepared_mutation(
        &mut self,
        prepared: PreparedProjectionMutation,
    ) -> Result<ProjectionRetirements, BaseWorkspaceError> {
        assert_eq!(self.revision, prepared.expected_revision);
        let mut view = self.ordered_view_mut()?;
        assert_eq!(view.pending().generation(), prepared.expected_generation);
        let ordered = &mut view.contents;
        let mut retired = ProjectionRetirements {
            storage: Some(std::mem::replace(
                &mut ordered.managed_storage,
                prepared.live,
            )),
            globals: Some(std::mem::take(&mut ordered.global_bindings)),
            ..ProjectionRetirements::default()
        };
        if prepared.captures_legacy {
            retired.entries = Some(std::mem::take(&mut ordered.entries));
        } else if let PreparedProjectionAction::Named { name } = prepared.action {
            retired.legacy_entry = ordered.entries.remove(name.as_str());
        }
        retired.delta_mutation = Some(ordered.pending.apply_prepared_mutation(prepared.pending));
        Ok(retired)
    }

    pub(super) fn plan_executor_write(
        &self,
        name: &str,
        value: &Value,
    ) -> Result<
        Option<WorkspaceExecutorWritePlan>,
        crate::context::host_allocation::HostAllocationError,
    > {
        let ordered = self.ordered_allocation_view()?;
        let plan = ordered.managed_storage().plan_executor_write(
            ordered
                .entries()
                .iter()
                .enumerate()
                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value())),
            name,
            value,
        )?;
        plan.map(|plan| {
            ordered
                .pending()
                .executor_assignment_retained_bytes(name)
                .and_then(|bytes| plan.with_additional_retained_bytes(bytes))
        })
        .transpose()
    }

    pub(super) fn plan_executor_retained_write(
        &self,
        name: &str,
        source: &WorkspaceSlotReadLease,
    ) -> Result<
        Option<WorkspaceExecutorWritePlan>,
        crate::context::host_allocation::HostAllocationError,
    > {
        let ordered = self.ordered_allocation_view()?;
        let plan = ordered.managed_storage().plan_executor_retained_write(
            ordered
                .entries()
                .iter()
                .enumerate()
                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value())),
            name,
            source,
        )?;
        plan.map(|plan| {
            ordered
                .pending()
                .executor_assignment_retained_bytes(name)
                .and_then(|bytes| plan.with_additional_retained_bytes(bytes))
        })
        .transpose()
    }

    pub(super) fn publication_upper_bound_after_executor(
        &self,
        name: &str,
    ) -> Result<usize, BaseWorkspaceError> {
        self.ordered_view()?
            .pending()
            .executor_change_len(name)
            .map_err(BaseWorkspaceError::HostAllocation)
    }

    pub(super) fn publication_upper_bound_after_appended_change(
        &self,
    ) -> Result<usize, BaseWorkspaceError> {
        self.ordered_view()?
            .pending()
            .appended_change_len()
            .map_err(BaseWorkspaceError::HostAllocation)
    }

    pub(super) fn publication_upper_bound(&self) -> Result<usize, BaseWorkspaceError> {
        Ok(self.ordered_view()?.pending().len())
    }

    pub(super) fn pending_publication(&self) -> Result<&WorkspaceDelta, BaseWorkspaceError> {
        Ok(self.ordered_view()?.pending())
    }

    pub(super) fn pending_generation(&self) -> u64 {
        match &self.contents {
            ProjectionContents::Ordered(ordered) => ordered.pending.generation(),
            ProjectionContents::Private(storage) => storage.epoch(),
        }
    }

    pub(super) fn plan_namespace_mutation(
        &self,
        name: &str,
    ) -> Result<WorkspaceNamespaceMutationPlan, crate::context::host_allocation::HostAllocationError>
    {
        let ordered = self.ordered_allocation_view()?;
        let plan = ordered.managed_storage().plan_namespace_mutation(
            ordered
                .entries()
                .iter()
                .enumerate()
                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value())),
            name,
        )?;
        let delta = ordered.pending().namespace_mutation_retained_bytes(name)?;
        plan.with_additional_retained_bytes(delta)
    }

    pub(super) fn prepare_namespace_mutation(
        &self,
        context: &crate::context::RuntimeContext,
        plan: WorkspaceNamespaceMutationPlan,
        name: WorkspaceName,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
        permit: &crate::context::host_allocation::HostOutputPermit,
    ) -> Result<PreparedProjectionMutation, BaseWorkspaceError> {
        let captures_legacy = plan.captures_legacy_entries();
        if !captures_legacy
            && self
                .ordered_view()?
                .contents
                .entries
                .contains_key(name.as_str())
        {
            return Err(BaseWorkspaceError::HostAllocation(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            ));
        }
        let entry_name = WorkspaceEntryName::Prepared(name.clone());
        let target = target.or_else(|| {
            (action == WorkspaceNamespaceAction::BindGlobal)
                .then(|| GlobalBindingTarget::Named(name.clone()))
        });
        let pending = match action {
            WorkspaceNamespaceAction::Remove => self
                .ordered_view()?
                .contents
                .pending
                .prepare_remove(entry_name, permit),
            WorkspaceNamespaceAction::BindGlobal => self
                .ordered_view()?
                .contents
                .pending
                .prepare_bind_global(entry_name, target.clone().expect("global target"), permit),
            WorkspaceNamespaceAction::UnbindGlobal => self
                .ordered_view()?
                .contents
                .pending
                .prepare_unbind_global(entry_name, permit),
        }
        .map_err(BaseWorkspaceError::HostAllocation)?;
        let mut live = self.ordered_view()?.contents.managed_storage.clone();
        live.prepare_namespace_mutation(
            plan,
            self.ordered_view()?
                .contents
                .entries
                .iter()
                .enumerate()
                .map(|(slot, (name, entry))| (slot, name.as_str(), entry.value())),
            name.clone(),
            action,
            target.clone(),
            permit,
        )
        .map_err(BaseWorkspaceError::HostAllocation)?;
        let live = live
            .prepare_live_with_globals(
                context,
                &self.ordered_view()?.contents.global_bindings,
                (action != WorkspaceNamespaceAction::Remove).then_some(name.as_str()),
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(PreparedProjectionMutation {
            live,
            pending: PreparedWorkspaceDeltaMutation::ReplaceDelta(pending),
            action: PreparedProjectionAction::Named { name },
            captures_legacy,
            expected_revision: self.revision,
            expected_generation: self.ordered_view()?.contents.pending.generation(),
        })
    }

    pub(super) fn plan_clear(
        &self,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        let ordered = self.ordered_allocation_view()?;
        ordered.pending().clear_retained_bytes()
    }

    pub(super) fn prepare_clear_mutation(
        &self,
        permit: &crate::context::host_allocation::HostOutputPermit,
    ) -> Result<PreparedProjectionMutation, BaseWorkspaceError> {
        let pending = self
            .ordered_view()?
            .contents
            .pending
            .prepare_clear(permit)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(PreparedProjectionMutation {
            live: ManagedWorkspaceStorage::empty_live(),
            pending: PreparedWorkspaceDeltaMutation::ReplaceDelta(pending),
            action: PreparedProjectionAction::Clear,
            captures_legacy: true,
            expected_revision: self.revision,
            expected_generation: self.ordered_view()?.contents.pending.generation(),
        })
    }

    /// Stage one source binding without retaining obsolete versions of that
    /// binding until the next executor safepoint. Explicit ordered delta and
    /// dynamic-effect entry points retain their separate replay semantics.
    #[cfg(test)]
    pub(super) fn assign_from_executor(
        &mut self,
        name: &str,
        value: Value,
    ) -> Result<(), BaseWorkspaceError> {
        drop(self.assign_unbounded_retired(name, value, None, false, true)?);
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<(), BaseWorkspaceError> {
        drop(self.namespace_unbounded_retired(name, WorkspaceNamespaceAction::Remove, None)?);
        Ok(())
    }

    pub(super) fn namespace_unbounded_retired(
        &mut self,
        name: &str,
        action: WorkspaceNamespaceAction,
        target: Option<&GlobalBindingTarget>,
    ) -> Result<ProjectionRetirements, BaseWorkspaceError> {
        self.capture_canonical_prefix(None)?;
        let prepared = self.prepare_unbounded_namespace(
            name,
            action,
            target,
            WorkspaceMutationOrigin::Ordered,
        )?;
        self.reserve_unbounded_mutation_capacity(&prepared)?;
        Ok(self
            .ordered_view_mut()?
            .install_unbounded_mutation(prepared))
    }

    pub fn clear(&mut self) -> Result<(), BaseWorkspaceError> {
        drop(self.clear_unbounded_retired()?);
        Ok(())
    }

    pub(super) fn clear_unbounded_retired(
        &mut self,
    ) -> Result<ProjectionRetirements, BaseWorkspaceError> {
        self.capture_canonical_prefix(None)?;
        let prepared = self.prepare_unbounded_clear(WorkspaceMutationOrigin::Ordered)?;
        self.reserve_unbounded_mutation_capacity(&prepared)?;
        Ok(self
            .ordered_view_mut()?
            .install_unbounded_mutation(prepared))
    }

    pub fn bind_global(&mut self, name: impl Into<String>) -> Result<(), BaseWorkspaceError> {
        let name = name.into();
        let target = GlobalBindingTarget::Named(WorkspaceName::unbounded(&name));
        self.bind_global_target(name, target)
    }

    pub fn bind_global_target(
        &mut self,
        name: impl Into<String>,
        target: GlobalBindingTarget,
    ) -> Result<(), BaseWorkspaceError> {
        let name = name.into();
        drop(self.namespace_unbounded_retired(
            &name,
            WorkspaceNamespaceAction::BindGlobal,
            Some(&target),
        )?);
        Ok(())
    }

    pub fn unbind_global(&mut self, name: impl Into<String>) -> Result<(), BaseWorkspaceError> {
        let name = name.into();
        drop(self.namespace_unbounded_retired(
            &name,
            WorkspaceNamespaceAction::UnbindGlobal,
            None,
        )?);
        Ok(())
    }

    /// Publish this projection's ordered mutations over the latest canonical
    /// state. A later publisher wins for names both projections changed; names
    /// absent from this delta are never overwritten.
    pub fn publish(&mut self) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let mut ordered = self.ordered_view_mut()?;
        let effect = ordered.pending().clone_unbounded();
        let (commit, retired) = ordered
            .publish_with_effect(effect, None)
            .map_err(|(error, _effect)| error)?;
        drop(retired);
        Ok(commit)
    }
}

impl OrderedProjectionMut<'_> {
    /// Install one independently owned canonical effect. The authentic
    /// pending delta remains in `self` until after the Base revision swaps;
    /// its ignored/intermediate Foreign values cannot retire during effect
    /// normalization or under either Base/service borrow.
    pub(in crate::workspace::base) fn publish_with_effect(
        &mut self,
        effect: WorkspaceDelta,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<(BaseWorkspaceCommit, ProjectionRetirements), (BaseWorkspaceError, WorkspaceDelta)>
    {
        let Some(owner) = self.owner.upgrade() else {
            return Err((BaseWorkspaceError::Closed, effect));
        };
        if self.contents.pending.is_empty() {
            let Ok(state) = owner.try_borrow() else {
                return Err((BaseWorkspaceError::BorrowConflict, effect));
            };
            if let Err(error) = state.preflight_commit(*self.revision, &self.contents.pending) {
                return Err((error, effect));
            }
            let commit =
                BaseWorkspaceCommit::new(*self.revision, state.revision, state.revision, effect);
            let mut retired = ProjectionRetirements::default();
            if state.entries.is_empty() && state.global_bindings.is_empty() {
                retired.storage = Some(std::mem::replace(
                    &mut self.contents.managed_storage,
                    state.managed_storage.clone(),
                ));
                retired.entries = Some(std::mem::take(&mut self.contents.entries));
                retired.globals = Some(std::mem::take(&mut self.contents.global_bindings));
                *self.revision = state.revision;
            }
            return Ok((commit, retired));
        }
        let (canonical_prefix, canonical_revision) = {
            let Ok(state) = owner.try_borrow_mut() else {
                return Err((BaseWorkspaceError::BorrowConflict, effect));
            };
            if let Err(error) = state.preflight_commit(*self.revision, &self.contents.pending) {
                return Err((error, effect));
            }
            (state.managed_storage.clone(), state.revision)
        };
        let mut retired = ProjectionRetirements {
            _projection_prefix: Some(self.contents.managed_storage.retained_published_prefix()),
            ..ProjectionRetirements::default()
        };
        // A complete prepared live tree already contains the admitted final
        // state when no semantic canonical write has interleaved. Revision
        // equality is required; shared allocation identity is not authority.
        let prepared_live = *self.revision == canonical_revision
            && self.contents.entries.is_empty()
            && self.contents.global_bindings.is_empty()
            && self.contents.managed_storage.is_prepared_live();
        let storage = match if prepared_live {
            Ok(self.contents.managed_storage.clone())
        } else {
            self.contents.managed_storage.prepare_publication_over(
                context,
                &canonical_prefix,
                &self.contents.pending,
            )
        } {
            Ok(storage) => storage,
            Err(error) => return Err((BaseWorkspaceError::HostAllocation(error), effect)),
        };
        let mut prepared = {
            let state = owner
                .try_borrow()
                .expect("validated Base cannot gain a borrow during synchronous publication");
            state
                .prepare_projection_commit(*self.revision, &self.contents.pending, effect, storage)
                .expect("validated Base revision cannot change without a callback")
        };
        prepared.canonicalize_effect();
        let (commit, canonical_storage, revision) = {
            let mut state = owner
                .try_borrow_mut()
                .expect("validated Base cannot gain a borrow during synchronous publication");
            state
                .ensure_open()
                .expect("validated Base cannot close without a callback");
            assert_eq!(state.revision, prepared.expected_revision());
            let (commit, old_storage) = state.install_direct_commit(prepared);
            // Keep the complete old canonical backing until both publication
            // borrows and the observer commit have finished.
            retired.storage = Some(old_storage);
            (commit, state.managed_storage.clone(), state.revision)
        };
        self.contents.managed_storage = canonical_storage;
        *self.revision = revision;
        retired.pending = Some(std::mem::take(&mut self.contents.pending));
        if !self.contents.entries.is_empty() {
            retired.entries = Some(std::mem::take(&mut self.contents.entries));
        }
        if !self.contents.global_bindings.is_empty() {
            retired.globals = Some(std::mem::take(&mut self.contents.global_bindings));
        }
        retired.canonical_prefix = Some(canonical_prefix);
        Ok((commit, retired))
    }
}

impl BaseWorkspaceProjection {
    pub(in crate::workspace::base) fn publish_with_effect(
        &mut self,
        effect: WorkspaceDelta,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<(BaseWorkspaceCommit, ProjectionRetirements), (BaseWorkspaceError, WorkspaceDelta)>
    {
        match self.ordered_view_mut() {
            Ok(mut ordered) => ordered.publish_with_effect(effect, context),
            Err(error) => Err((error, effect)),
        }
    }

    /// Publish a direct-owner completion using its independently prepared
    /// effect backing. No output allocation or source-value clone occurs at
    /// the terminal publication boundary.
    pub(super) fn publish_terminal_with_effect(
        &mut self,
        effect: WorkspaceDelta,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        let (commit, retired) = self
            .publish_with_effect(effect, context)
            .map_err(|(error, _effect)| error)?;
        drop(retired);
        Ok(commit)
    }

    /// Refresh a clean projection from the canonical owner. Dirty projections
    /// must publish first so event order, rather than snapshot replacement,
    /// decides same-name writes.
    pub fn reconcile(&mut self) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        if self.has_pending_changes() {
            return Err(BaseWorkspaceError::PendingChanges);
        }
        let owner = self.owner.upgrade().ok_or(BaseWorkspaceError::Closed)?;
        let (reconciliation, managed_storage) = {
            let state = owner
                .try_borrow()
                .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
            let mut reconciliation = state.reconcile(self.revision)?;
            // A stable physical capture can migrate the canonical owner's
            // legacy maps into a managed generation without advancing its
            // semantic revision. Whether the semantic reconciliation is
            // UpToDate or Delta, retaining this projection's old legacy maps
            // beside that generation would duplicate unrelated bindings.
            if state
                .managed_storage
                .is_canonical_without_legacy(&state.entries, &state.global_bindings)
                && (!self.ordered_view()?.contents.entries.is_empty()
                    || !self.ordered_view()?.contents.global_bindings.is_empty())
            {
                reconciliation = BaseWorkspaceReconciliation::Reset(state.snapshot());
            }
            (reconciliation, state.managed_storage.clone())
        };
        self.apply_reconciliation(&reconciliation, &managed_storage)?;
        Ok(reconciliation)
    }

    fn apply_reconciliation(
        &mut self,
        reconciliation: &BaseWorkspaceReconciliation,
        managed_storage: &ManagedWorkspaceStorage,
    ) -> Result<(), BaseWorkspaceError> {
        let mut view = self.ordered_view_mut()?;
        match reconciliation {
            BaseWorkspaceReconciliation::UpToDate { revision } => {
                view.contents.managed_storage = managed_storage.clone();
                *view.revision = *revision;
            }
            BaseWorkspaceReconciliation::Delta { to, changes, .. } => {
                let ordered = &mut view.contents;
                ordered.managed_storage = managed_storage.clone();
                apply_delta(
                    &mut ordered.entries,
                    &mut ordered.global_bindings,
                    &ordered.managed_storage,
                    changes,
                );
                *view.revision = *to;
            }
            BaseWorkspaceReconciliation::Reset(snapshot) => {
                view.contents.entries = snapshot.entries.clone();
                view.contents.global_bindings = snapshot.global_bindings.clone();
                view.contents.managed_storage = snapshot.managed_storage.clone();
                *view.revision = snapshot.revision();
            }
        }
        Ok(())
    }
}

fn apply_delta(
    entries: &mut RootedWorkspaceEntries,
    global_bindings: &mut HashMap<String, GlobalBindingTarget>,
    managed_storage: &ManagedWorkspaceStorage,
    delta: &WorkspaceDelta,
) {
    for change in delta.changes() {
        match change.data() {
            WorkspaceChangeData::Assign { name, entry, .. } => {
                if change.storage() == WorkspaceChangeStorage::ManagedOverlay
                    && managed_storage.lookup(name.as_str()).is_some()
                {
                    global_bindings.remove(name.as_str());
                    entries.remove(name.as_str());
                } else if managed_storage.global(name.as_str()) != Some(true)
                    && !global_bindings.contains_key(name.as_str())
                {
                    entries.insert(name.clone(), entry.clone());
                }
            }
            WorkspaceChangeData::Remove { name } => {
                entries.remove(name.as_str());
            }
            WorkspaceChangeData::BindGlobal { name, target } => {
                entries.remove(name.as_str());
                if change.storage() == WorkspaceChangeStorage::Legacy {
                    global_bindings.insert(name.as_str().to_string(), target.clone());
                } else {
                    // The canonical managed generation owns this exact target.
                    // Retire a prior legacy binding rather than keeping a
                    // second, stale authority for the same workspace name.
                    global_bindings.remove(name.as_str());
                }
            }
            WorkspaceChangeData::UnbindGlobal { name } => {
                entries.remove(name.as_str());
                global_bindings.remove(name.as_str());
            }
            WorkspaceChangeData::Clear => {
                entries.clear();
                global_bindings.clear();
            }
        }
    }
}
