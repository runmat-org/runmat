use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use runmat_value::Value;

use super::journal::{WorkspaceJournal, DEFAULT_HISTORY_CHANGE_LIMIT};
use super::storage::{ManagedWorkspaceStorage, PrivateLiveStorage};
use super::values::RootedWorkspaceEntries;
use super::{BaseWorkspaceError, BaseWorkspaceProjection, WorkspaceDelta, WorkspaceRevision};
use crate::workspace::GlobalBindingTarget;
use crate::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};
use crate::workspace::{WorkspaceCompletionIdentity, WorkspaceCompletionSource};

#[derive(Clone, Debug)]
pub struct BaseWorkspaceSnapshot {
    revision: WorkspaceRevision,
    pub(super) entries: RootedWorkspaceEntries,
    pub(super) global_bindings: HashMap<String, GlobalBindingTarget>,
    pub(super) managed_storage: ManagedWorkspaceStorage,
}

impl BaseWorkspaceSnapshot {
    pub(super) fn new(
        revision: WorkspaceRevision,
        entries: RootedWorkspaceEntries,
        global_bindings: HashMap<String, GlobalBindingTarget>,
        managed_storage: ManagedWorkspaceStorage,
    ) -> Self {
        Self {
            revision,
            entries,
            global_bindings,
            managed_storage,
        }
    }

    pub(super) fn from_managed_projection(
        revision: WorkspaceRevision,
        managed_storage: ManagedWorkspaceStorage,
    ) -> Self {
        Self {
            revision,
            entries: RootedWorkspaceEntries::default(),
            global_bindings: HashMap::new(),
            managed_storage,
        }
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.managed_storage.iter().count() + self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.managed_storage.iter().next().is_none() && self.entries.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.managed_storage
            .lookup(name)
            .or_else(|| self.entries.get(name).map(|entry| entry.value()))
    }

    pub(crate) fn read_lease(
        &self,
        name: &str,
    ) -> Option<crate::workspace::WorkspaceValueReadLease> {
        self.managed_storage.read_lease(name).or_else(|| {
            self.entries
                .get(name)
                .map(super::values::RootedWorkspaceEntry::read_lease)
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.managed_storage.iter().chain(
            self.entries
                .iter()
                .map(|(name, entry)| (name.as_str(), entry.value())),
        )
    }

    pub fn is_global(&self, name: &str) -> bool {
        self.managed_storage
            .global(name)
            .unwrap_or_else(|| self.global_bindings.contains_key(name))
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        match self.managed_storage.global(name) {
            Some(true) => self.managed_storage.global_target(name),
            Some(false) => None,
            None => self.global_bindings.get(name),
        }
    }

    pub fn global_names(&self) -> impl Iterator<Item = &str> {
        self.managed_storage.global_names().chain(
            self.global_bindings
                .keys()
                .map(|name| name.as_str())
                .filter(|name| self.managed_storage.global(name).is_none()),
        )
    }

    pub(in crate::workspace) fn fork_managed_owner(
        &self,
    ) -> Result<BaseWorkspaceOwner, BaseWorkspaceError> {
        if !self.entries.is_empty() || !self.global_bindings.is_empty() {
            return Err(BaseWorkspaceError::HostAllocation(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            ));
        }
        let mut journal = WorkspaceJournal::new(DEFAULT_HISTORY_CHANGE_LIMIT);
        journal.clear_at(self.revision);
        Ok(BaseWorkspaceOwner {
            state: Rc::new(RefCell::new(BaseWorkspaceState {
                closed: false,
                revision: self.revision,
                entries: RootedWorkspaceEntries::new(),
                global_bindings: HashMap::new(),
                journal,
                managed_storage: self.managed_storage.clone(),
                private_live: None,
            })),
        })
    }

    /// Materialize independent language values for an explicit host/export
    /// boundary. Executor admission should use [`BaseWorkspaceOwner::begin_projection`]
    /// so unchanged entries remain shared.
    pub fn to_values(&self) -> Vec<(String, Value)> {
        self.iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    }
}

#[derive(Debug)]
pub struct BaseWorkspaceCommit {
    base_revision: WorkspaceRevision,
    previous_revision: WorkspaceRevision,
    revision: WorkspaceRevision,
    changes: WorkspaceDelta,
}

impl BaseWorkspaceCommit {
    pub(super) fn new(
        base_revision: WorkspaceRevision,
        previous_revision: WorkspaceRevision,
        revision: WorkspaceRevision,
        changes: WorkspaceDelta,
    ) -> Self {
        Self {
            base_revision,
            previous_revision,
            revision,
            changes,
        }
    }

    pub fn base_revision(&self) -> WorkspaceRevision {
        self.base_revision
    }

    pub fn previous_revision(&self) -> WorkspaceRevision {
        self.previous_revision
    }

    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Canonical effect after ordered input changes were applied. Intermediate
    /// assigned values that were overwritten, removed, or cleared are absent.
    pub fn changes(&self) -> &WorkspaceDelta {
        &self.changes
    }

    pub fn merged_concurrent_changes(&self) -> bool {
        self.base_revision != self.previous_revision
    }
}

#[derive(Debug)]
pub enum BaseWorkspaceReconciliation {
    UpToDate {
        revision: WorkspaceRevision,
    },
    Delta {
        from: WorkspaceRevision,
        to: WorkspaceRevision,
        changes: WorkspaceDelta,
    },
    Reset(BaseWorkspaceSnapshot),
}

#[derive(Clone, Debug)]
pub struct BaseWorkspaceOwner {
    pub(super) state: Rc<RefCell<BaseWorkspaceState>>,
}

#[derive(Debug)]
pub(super) struct BaseWorkspaceState {
    pub(super) closed: bool,
    pub(super) revision: WorkspaceRevision,
    pub(super) entries: RootedWorkspaceEntries,
    pub(super) global_bindings: HashMap<String, GlobalBindingTarget>,
    pub(super) journal: WorkspaceJournal,
    pub(super) managed_storage: ManagedWorkspaceStorage,
    pub(super) private_live: Option<PrivateLiveStorage>,
}

impl Default for BaseWorkspaceOwner {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseWorkspaceOwner {
    /// Whether two service ports retain the same canonical workspace owner.
    pub fn same_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
    }

    pub(super) fn from_private_storage(storage: PrivateLiveStorage) -> Self {
        let revision = storage.revision();
        let mut journal = WorkspaceJournal::new(DEFAULT_HISTORY_CHANGE_LIMIT);
        journal.clear_at(revision);
        Self {
            state: Rc::new(RefCell::new(BaseWorkspaceState {
                closed: false,
                revision,
                entries: RootedWorkspaceEntries::new(),
                global_bindings: HashMap::new(),
                journal,
                managed_storage: ManagedWorkspaceStorage::default(),
                private_live: Some(storage),
            })),
        }
    }

    /// Apply one normalized dynamic-target completion directly to this owner.
    /// The exact identity is revalidated before preparation and immediately
    /// before publication; no action is installed through a scalar prefix.
    pub fn apply_workspace_completion(
        &self,
        context: &crate::context::RuntimeContext,
        identity: WorkspaceCompletionIdentity,
        source: &dyn WorkspaceCompletionSource,
    ) -> Result<(), BaseWorkspaceError> {
        let issuer = Rc::as_ptr(&self.state) as usize;
        let validate = || {
            let state = self
                .state
                .try_borrow()
                .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
            state.ensure_open()?;
            if identity.issuer() != issuer
                || identity.revision() != state.revision
                || identity.pending_generation() != 0
            {
                return Err(BaseWorkspaceError::CompletionConflict);
            }
            Ok(())
        };
        validate()?;
        if source.is_empty() {
            let plan = crate::context::host_allocation::HostAllocationPlan::new(0, 0)
                .map_err(BaseWorkspaceError::HostAllocation)?;
            if let Some(receipt) =
                crate::context::host_allocation::HostAllocationReceipt::admit(Some(context), plan)
                    .map_err(BaseWorkspaceError::HostAllocation)?
            {
                receipt
                    .checkpoint(context)
                    .map_err(BaseWorkspaceError::HostAllocation)?;
            }
            return Ok(());
        }
        // Prepare the canonical physical prefix before selecting the
        // completion route. The terminal publication must never discover a
        // legacy map migration after the source effect or cancellation.
        self.capture_stable_with_context(context)?;
        let policy_probe = crate::context::host_allocation::HostAllocationPlan::new(0, 0)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let finite = crate::context::host_allocation::HostAllocationReceipt::admit(
            Some(context),
            policy_probe,
        )
        .map_err(BaseWorkspaceError::HostAllocation)?
        .is_some();
        let mut projection = if finite {
            self.begin_managed_completion_projection()?
        } else {
            self.begin_projection()?
        };
        let (retired, effect) = if finite {
            let retained = projection
                .completion_retained_bytes(source)
                .map_err(BaseWorkspaceError::HostAllocation)?;
            let plan = crate::context::host_allocation::HostAllocationPlan::new(0, retained)
                .map_err(BaseWorkspaceError::HostAllocation)?;
            match crate::context::host_allocation::HostAllocationReceipt::admit(Some(context), plan)
                .map_err(BaseWorkspaceError::HostAllocation)?
            {
                Some(receipt) => {
                    let permit = receipt
                        .output_permit(context)
                        .map_err(BaseWorkspaceError::HostAllocation)?;
                    let completion = projection.prepare_completion(context, source, &permit)?;
                    // The completion quote includes this independent effect
                    // buffer. Fund it before the checkpoint; publication may
                    // not allocate after the language-visible completion.
                    let effect = completion
                        .prepare_effect(&permit)
                        .map_err(BaseWorkspaceError::HostAllocation)?;
                    receipt
                        .checkpoint(context)
                        .map_err(BaseWorkspaceError::HostAllocation)?;
                    validate()?;
                    (
                        projection
                            .ordered_view_mut()?
                            .install_completion(completion),
                        effect,
                    )
                }
                None => {
                    let completion = projection.prepare_completion_unbounded(source)?;
                    let effect = completion.prepare_effect_unbounded();
                    validate()?;
                    (
                        projection
                            .ordered_view_mut()?
                            .install_completion(completion),
                        effect,
                    )
                }
            }
        } else {
            let completion = projection.prepare_completion_unbounded(source)?;
            let effect = completion.prepare_effect_unbounded();
            validate()?;
            (
                projection
                    .ordered_view_mut()?
                    .install_completion(completion),
                effect,
            )
        };
        let publication = projection.publish_terminal_with_effect(effect, Some(context));
        drop(retired);
        publication?;
        Ok(())
    }

    /// A finite direct-owner completion may only clone immutable managed
    /// owners. Migrating legacy maps into managed storage is a separate
    /// allocation-bearing operation and cannot be hidden before its receipt.
    pub(in crate::workspace::base) fn begin_managed_completion_projection(
        &self,
    ) -> Result<BaseWorkspaceProjection, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if !state.entries.is_empty() || !state.global_bindings.is_empty() {
            return Err(BaseWorkspaceError::HostAllocation(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                ),
            ));
        }
        drop(state);
        self.begin_projection()
    }

    /// One canonical selection-and-retirement section. The returned old
    /// values are independently rooted before the revision is published.
    pub fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, BaseWorkspaceError> {
        self.capture_stable_unbounded()?;
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        let mut values: Vec<_> =
            state
                .managed_storage
                .iter()
                .filter(|(name, _)| {
                    !state
                        .managed_storage
                        .global(name)
                        .unwrap_or_else(|| state.global_bindings.contains_key(*name))
                })
                .map(|(name, value)| (name.to_owned(), value.clone()))
                .chain(
                    state
                        .entries
                        .iter()
                        .filter(|(name, _)| {
                            state.managed_storage.lookup(name.as_str()).is_none()
                                && !state.managed_storage.global(name.as_str()).unwrap_or_else(
                                    || state.global_bindings.contains_key(name.as_str()),
                                )
                        })
                        .map(|(name, entry)| (name.as_str().to_owned(), entry.clone_value())),
                )
                .collect();
        let targets: Vec<_> = state
            .managed_storage
            .global_names()
            .filter_map(|name| {
                state
                    .managed_storage
                    .global_target(name)
                    .cloned()
                    .map(|target| (name.to_owned(), target))
            })
            .chain(
                state
                    .global_bindings
                    .iter()
                    .filter(|(name, _)| state.managed_storage.global(name.as_str()).is_none())
                    .map(|(name, target)| (name.clone(), target.clone())),
            )
            .collect();
        let base = state.revision;
        drop(state);
        let mut delta = WorkspaceDelta::new();
        match selection {
            WorkspaceDetachSelection::All => delta.clear(),
            WorkspaceDetachSelection::Names(names) => {
                for name in names {
                    if targets.iter().any(|(global, _)| global == name) {
                        delta.unbind_global(name.clone());
                    } else {
                        delta.remove(name.clone());
                    }
                }
            }
            WorkspaceDetachSelection::Except(excluded) => {
                let excluded: std::collections::HashSet<_> =
                    excluded.iter().map(String::as_str).collect();
                for (name, _) in &values {
                    if !excluded.contains(name.as_str()) {
                        delta.remove(name.clone());
                    }
                }
                for (name, _) in &targets {
                    if !excluded.contains(name.as_str()) {
                        delta.unbind_global(name.clone());
                    }
                }
            }
        }
        for (name, target) in &targets {
            let value = match target {
                GlobalBindingTarget::Named(target) => {
                    crate::workspace::session::global_value(target.as_str())
                }
                GlobalBindingTarget::LegacySlot(slot) => {
                    crate::workspace::session::global_slot_value(*slot)
                }
            };
            if let Some(value) = value {
                values.push((name.clone(), value));
            }
        }
        let detached = DetachedWorkspaceValues::new(selection.select(values))?;
        drop(self.commit_direct(base, delta)?);
        Ok(detached)
    }

    pub fn new() -> Self {
        Self::with_history_limit(DEFAULT_HISTORY_CHANGE_LIMIT)
    }

    pub fn snapshot(&self) -> Result<BaseWorkspaceSnapshot, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        Ok(state.snapshot())
    }

    /// Capture a stable generation for an executor without copying workspace
    /// maps or Values. Legacy entries are migrated once, under this operation's
    /// host admission; subsequent captures clone only immutable owner handles.
    pub fn capture_stable_with_context(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<BaseWorkspaceSnapshot, BaseWorkspaceError> {
        self.capture_stable(Some(context))
    }

    /// The explicit unbounded compatibility route uses the same physical
    /// capture algorithm, but does not mint a managed receipt.
    pub(super) fn capture_stable_unbounded(
        &self,
    ) -> Result<BaseWorkspaceSnapshot, BaseWorkspaceError> {
        self.capture_stable(None)
    }

    fn capture_stable(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> Result<BaseWorkspaceSnapshot, BaseWorkspaceError> {
        use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};

        // A canonical generation requires no new allocation, but capture is
        // still an operation against the current cancellation/ledger epoch.
        HostAllocationReceipt::policy(context).map_err(BaseWorkspaceError::HostAllocation)?;

        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        if state
            .managed_storage
            .is_canonical_without_legacy(&state.entries, &state.global_bindings)
        {
            return Ok(state.snapshot());
        }
        let revision = state.revision;
        let plan = state
            .managed_storage
            .plan_capture(&state.entries, &state.global_bindings)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let receipt = HostAllocationReceipt::admit(
            context,
            HostAllocationPlan::new(0, plan.retained_bytes())
                .map_err(BaseWorkspaceError::HostAllocation)?,
        )
        .map_err(BaseWorkspaceError::HostAllocation)?;
        let permit = receipt
            .as_ref()
            .map(|receipt| receipt.output_permit(context.expect("managed receipt has context")))
            .transpose()
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let prepared = state
            .managed_storage
            .prepare_capture(
                plan,
                &state.entries,
                &state.global_bindings,
                permit.as_ref(),
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context.expect("managed receipt has context"))
                .map_err(BaseWorkspaceError::HostAllocation)?;
        }
        drop(state);

        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.revision != revision {
            return Err(BaseWorkspaceError::CompletionConflict);
        }
        let old_entries = std::mem::take(&mut state.entries);
        let old_globals = std::mem::take(&mut state.global_bindings);
        let old_storage = std::mem::replace(&mut state.managed_storage, prepared);
        let snapshot = state.snapshot();
        drop(state);
        drop(old_entries);
        drop(old_globals);
        drop(old_storage);
        Ok(snapshot)
    }

    pub fn lookup(&self, name: &str) -> Result<Option<Value>, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        let is_global = state
            .managed_storage
            .global(name)
            .unwrap_or_else(|| state.global_bindings.contains_key(name));
        Ok((!is_global)
            .then(|| {
                state
                    .managed_storage
                    .lookup(name)
                    .cloned()
                    .or_else(|| state.entries.get(name).map(|entry| entry.clone_value()))
            })
            .flatten())
    }

    pub fn is_global(&self, name: &str) -> Result<bool, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        Ok(state
            .managed_storage
            .global(name)
            .unwrap_or_else(|| state.global_bindings.contains_key(name)))
    }

    pub fn global_target(
        &self,
        name: &str,
    ) -> Result<Option<GlobalBindingTarget>, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        Ok(match state.managed_storage.global(name) {
            Some(true) => state.managed_storage.global_target(name),
            Some(false) => None,
            None => state.global_bindings.get(name),
        }
        .cloned())
    }

    pub fn global_names(&self) -> Result<Vec<String>, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        let mut names = state
            .managed_storage
            .global_names()
            .map(str::to_string)
            .chain(
                state
                    .global_bindings
                    .iter()
                    .filter(|(name, _)| state.managed_storage.global(name).is_none())
                    .map(|(name, _)| name.clone()),
            )
            .collect::<Vec<_>>();
        names.sort();
        Ok(names)
    }

    /// Visit canonical retained values without materializing a workspace
    /// snapshot. `None` means the visitor stopped before completing the view.
    pub fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Result<Option<()>, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        for (_, value) in state.managed_storage.iter().chain(
            state
                .entries
                .iter()
                .map(|(name, entry)| (name.as_str(), entry.value())),
        ) {
            if visitor(value).is_none() {
                return Ok(None);
            }
        }
        Ok(Some(()))
    }

    pub fn revision(&self) -> Result<WorkspaceRevision, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        Ok(state.revision)
    }

    pub fn workspace_completion_identity(
        &self,
    ) -> Result<crate::workspace::WorkspaceCompletionIdentity, BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()?;
        if state.private_live.is_some() {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        Ok(crate::workspace::WorkspaceCompletionIdentity::issued(
            state.revision,
            0,
            Rc::as_ptr(&self.state) as usize,
        ))
    }

    /// Confirm that this owner can still authorize a newly admitted external
    /// effect without materializing or exposing its workspace contents.
    pub fn ensure_effect_authority(&self) -> Result<(), BaseWorkspaceError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        state.ensure_open()
    }

    pub fn begin_projection(&self) -> Result<BaseWorkspaceProjection, BaseWorkspaceError> {
        if self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?
            .private_live
            .is_some()
        {
            return Err(BaseWorkspaceError::PrivateProjectionRequired);
        }
        BaseWorkspaceProjection::new(self)
    }

    pub fn commit(
        &self,
        base_revision: WorkspaceRevision,
        delta: WorkspaceDelta,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        self.commit_direct(base_revision, delta)
    }

    /// Apply a synchronous host mutation against the current canonical
    /// revision. Executor projections use [`Self::commit`] so their older base
    /// revision remains visible in the receipt.
    pub fn apply(&self, delta: WorkspaceDelta) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        self.capture_stable_unbounded()?;
        let state = self
            .state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        let revision = state.revision;
        drop(state);
        self.commit_direct(revision, delta)
    }

    fn commit_direct(
        &self,
        base_revision: WorkspaceRevision,
        original: WorkspaceDelta,
    ) -> Result<BaseWorkspaceCommit, BaseWorkspaceError> {
        self.capture_stable_unbounded()?;
        let mut prepared = {
            let state = self
                .state
                .try_borrow()
                .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
            state.prepare_direct_commit(base_revision, &original)?
        };
        // The original rooted input remains owned outside every Base borrow
        // while ignored/intermediate effects are normalized off-state.
        prepared.canonicalize_effect();
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        if let Err(error) = state.ensure_open() {
            drop(state);
            return Err(error);
        }
        if state.revision != prepared.expected_revision() {
            drop(state);
            return Err(BaseWorkspaceError::CompletionConflict);
        }
        let (commit, retired_storage) = state.install_direct_commit(prepared);
        drop(state);
        drop(retired_storage);
        drop(original);
        Ok(commit)
    }

    pub fn reconcile(
        &self,
        revision: WorkspaceRevision,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        self.state
            .try_borrow()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?
            .reconcile(revision)
    }

    pub fn close(&self) -> Result<bool, BaseWorkspaceError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| BaseWorkspaceError::BorrowConflict)?;
        if state.closed {
            return Ok(false);
        }
        state.closed = true;
        let retired_entries = std::mem::take(&mut state.entries);
        let retired_globals = std::mem::take(&mut state.global_bindings);
        let retired_storage = std::mem::take(&mut state.managed_storage);
        let retired_private = state.private_live.take();
        let revision = state.revision;
        state.journal.clear_at(revision);
        drop(state);
        drop(retired_entries);
        drop(retired_globals);
        drop(retired_storage);
        drop(retired_private);
        Ok(true)
    }

    pub fn is_closed(&self) -> Result<bool, BaseWorkspaceError> {
        self.state
            .try_borrow()
            .map(|state| state.closed)
            .map_err(|_| BaseWorkspaceError::BorrowConflict)
    }

    fn with_history_limit(history_change_limit: usize) -> Self {
        Self {
            state: Rc::new(RefCell::new(BaseWorkspaceState {
                closed: false,
                revision: WorkspaceRevision::INITIAL,
                entries: RootedWorkspaceEntries::new(),
                global_bindings: HashMap::new(),
                journal: WorkspaceJournal::new(history_change_limit),
                managed_storage: ManagedWorkspaceStorage::default(),
                private_live: None,
            })),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(history_change_limit: usize) -> Self {
        Self::with_history_limit(history_change_limit)
    }

    #[cfg(test)]
    pub(super) fn set_revision_for_test(&self, revision: WorkspaceRevision) {
        let mut state = self.state.borrow_mut();
        state.revision = revision;
        state.journal.clear_at(revision);
    }
}

impl BaseWorkspaceState {
    pub(super) fn ensure_open(&self) -> Result<(), BaseWorkspaceError> {
        if self.closed {
            Err(BaseWorkspaceError::Closed)
        } else {
            Ok(())
        }
    }

    pub(super) fn snapshot(&self) -> BaseWorkspaceSnapshot {
        BaseWorkspaceSnapshot::new(
            self.revision,
            self.entries.clone(),
            self.global_bindings.clone(),
            self.managed_storage.clone(),
        )
    }

    pub(super) fn reconcile(
        &self,
        revision: WorkspaceRevision,
    ) -> Result<BaseWorkspaceReconciliation, BaseWorkspaceError> {
        self.ensure_open()?;
        self.journal.reconcile(
            revision,
            self.revision,
            &self.entries,
            &self.global_bindings,
            &self.managed_storage,
        )
    }
}
