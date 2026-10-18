//! Prepared, move-only lexical half of a workspace completion.
//!
//! The source leases retain immutable Value generations and their independent
//! GC roots. Every table and action block is admitted before the short GC
//! mutation guard; completion then swaps owners without cloning a Value,
//! allocating, invoking a callback, or releasing a retired Foreign value.

use std::alloc::Layout;

use runmat_execution::cancellation::CancellationToken;
use runmat_gc::{
    gc_lock_lexical_mutations, gc_with_lexical_activation, GcHandle, LexicalMutationGuard,
    PreparedLexicalTable, RetiredLexicalTable,
};
use runmat_types::BindingId;
use runmat_value::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit, LexicalEnvironment,
    SharedValueGeneration,
};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::{RuntimeContext, RuntimeContextLocalId};
use crate::execution::RootedValueLease;
use crate::runtime_error::semantic_error;
use crate::workspace::{WorkspaceCompletionParticipant, WorkspaceSlotReadLease};
use crate::RuntimeError;

use super::environment::{environment_gc_error, environment_host_error, root_admission_error};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LexicalCompletionScope {
    Owned,
    Inherited,
}

/// The exact activation generation observed by a suspended lexical caller.
/// The activation root prevents handle reuse while the stamp is retained;
/// the revision detects any intervening binding mutation, including one that
/// restores the same Value. This does not detach or clone a language Value.
#[derive(Clone, Debug)]
pub struct LexicalBindingRevision {
    environment: LexicalEnvironment,
    binding: BindingId,
    scope: LexicalCompletionScope,
    activation: GcHandle,
    revision: u64,
    context: RuntimeContextLocalId,
    cancellation: CancellationToken,
    _root: RootedValueLease,
}

impl LexicalBindingRevision {
    pub fn capture_with_context(
        environment: &LexicalEnvironment,
        binding: BindingId,
        scope: LexicalCompletionScope,
        context: &RuntimeContext,
    ) -> Result<Self, RuntimeError> {
        HostAllocationReceipt::policy(Some(context)).map_err(environment_host_error)?;
        let activation = resolve_activation(environment, binding, scope)?;
        let root = RootedValueLease::for_handle_in_context(activation, context)
            .map_err(root_admission_error)?;
        let revision = gc_with_lexical_activation(&activation, |owner| owner.revision())
            .map_err(environment_gc_error)?;
        HostAllocationReceipt::policy(Some(context)).map_err(environment_host_error)?;
        Ok(Self {
            environment: environment.clone(),
            binding,
            scope,
            activation,
            revision,
            context: context.local_identity(),
            cancellation: context.cancellation(),
            _root: root,
        })
    }

    pub fn validate_with_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.context != context.local_identity()
            || !self.cancellation.same_generation(&context.cancellation())
        {
            return Err(environment_host_error(HostAllocationError::StaleAuthority));
        }
        HostAllocationReceipt::policy(Some(context)).map_err(environment_host_error)?;
        if resolve_activation(&self.environment, self.binding, self.scope)? != self.activation {
            return Err(lexical_error("lexical binding owner changed since capture"));
        }
        let revision = gc_with_lexical_activation(&self.activation, |owner| owner.revision())
            .map_err(environment_gc_error)?;
        if revision != self.revision {
            return Err(lexical_error("lexical activation changed since capture"));
        }
        Ok(())
    }
}

/// One final source action. `Assign` carries the already-retained source
/// generation; a borrowed `&Value` is deliberately not sufficient.
pub enum LexicalCompletionChange {
    Assign {
        environment: LexicalEnvironment,
        binding: BindingId,
        scope: LexicalCompletionScope,
        source: WorkspaceSlotReadLease,
    },
    Remove {
        environment: LexicalEnvironment,
        binding: BindingId,
        scope: LexicalCompletionScope,
    },
}

impl LexicalCompletionChange {
    fn parts(
        self,
    ) -> (
        LexicalEnvironment,
        BindingId,
        LexicalCompletionScope,
        Option<WorkspaceSlotReadLease>,
    ) {
        match self {
            Self::Assign {
                environment,
                binding,
                scope,
                source,
            } => (environment, binding, scope, Some(source)),
            Self::Remove {
                environment,
                binding,
                scope,
            } => (environment, binding, scope, None),
        }
    }
}

/// Exact Vec backing and its independent physical claim. Entries are retired
/// in source order, then the Vec allocation retires before its claim refunds.
struct ClaimedBacking<T> {
    values: Vec<T>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl<T> ClaimedBacking<T> {
    fn requested_bytes(capacity: usize) -> Result<usize, RuntimeError> {
        Layout::array::<T>(capacity)
            .map(|layout| layout.size())
            .map_err(|_| lexical_error("lexical completion layout overflow"))
    }

    fn prepare(
        capacity: usize,
        permit: Option<&dyn ExactHostAllocationPermit>,
    ) -> Result<Self, RuntimeError> {
        if capacity == 0 {
            return Ok(Self {
                values: Vec::new(),
                _claim: None,
            });
        }
        let bytes = Self::requested_bytes(capacity)?;
        let claim = permit
            .map(|permit| permit.reserve_exact_claim(bytes))
            .transpose()
            .map_err(|error| lexical_error(&error.to_string()))?;
        let values =
            exact_capacity_vec(capacity).map_err(|error| lexical_error(&error.to_string()))?;
        Ok(Self {
            values,
            _claim: claim,
        })
    }
}

impl<T> Drop for ClaimedBacking<T> {
    fn drop(&mut self) {
        for value in self.values.drain(..) {
            drop(value);
        }
    }
}

struct BatchAction {
    activation: GcHandle,
    group: usize,
    binding: BindingId,
    next: Option<SharedValueGeneration>,
    retired: Option<SharedValueGeneration>,
    // Physical generations retire before these external root leases. The
    // prior lease protects a replaced value after GC stops tracing its old
    // activation slot and before the old owner is finally released.
    _source: Option<WorkspaceSlotReadLease>,
    _prior_roots: Option<RootedValueLease>,
}

struct ActivationGroup {
    handle: GcHandle,
    _root: RootedValueLease,
    expected_revision: u64,
    next_revision: u64,
    missing: usize,
    prepared_table: Option<PreparedLexicalTable>,
    table_receipt: Option<HostAllocationReceipt>,
    retired_table: Option<RetiredLexicalTable>,
}

pub struct PreparedLexicalCompletionBatch {
    actions: ClaimedBacking<BatchAction>,
    groups: ClaimedBacking<ActivationGroup>,
    handles: ClaimedBacking<GcHandle>,
    keys: ClaimedBacking<(usize, BindingId, usize)>,
    receipt: Option<HostAllocationReceipt>,
}

impl PreparedLexicalCompletionBatch {
    /// Commit a lexical-only completion with its prepared source frame. Both
    /// fallible checkpoints precede any installation; each short install is
    /// allocation-free and callback-free. Old lexical and frame owners retire
    /// only after the GC mutation guard has been released.
    pub fn commit_with_participant<P>(
        &mut self,
        context: &RuntimeContext,
        participant: &mut P,
    ) -> Result<P::Retirement, RuntimeError>
    where
        P: WorkspaceCompletionParticipant,
    {
        participant.checkpoint_before_install(context)?;
        let ready = self.checkpoint(context)?;
        let (retired_lexical, retired_frame) = ready.commit_with_participant(participant);
        drop(retired_lexical);
        Ok(retired_frame)
    }

    pub fn prepare(
        context: &RuntimeContext,
        changes: impl ExactSizeIterator<Item = LexicalCompletionChange>,
    ) -> Result<Self, RuntimeError> {
        let count = changes.len();
        let action_bytes = ClaimedBacking::<BatchAction>::requested_bytes(count)?;
        let group_bytes = ClaimedBacking::<ActivationGroup>::requested_bytes(count)?;
        let handle_bytes = ClaimedBacking::<GcHandle>::requested_bytes(count)?;
        let key_bytes = ClaimedBacking::<(usize, BindingId, usize)>::requested_bytes(count)?;
        let raw = action_bytes
            .checked_add(group_bytes)
            .and_then(|bytes| bytes.checked_add(handle_bytes))
            .and_then(|bytes| bytes.checked_add(key_bytes))
            .ok_or_else(|| lexical_error("lexical completion layout overflow"))?;
        let raw =
            u64::try_from(raw).map_err(|_| lexical_error("lexical completion layout overflow"))?;
        let claim_count = if count == 0 { 0 } else { 4 };
        let quote = HostAllocationPlan::retained_exact_output_bytes(raw, claim_count)
            .map_err(environment_host_error)?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, quote).map_err(environment_host_error)?,
        )
        .map_err(environment_host_error)?;
        let permit = receipt
            .as_ref()
            .map(|receipt| receipt.output_permit(context))
            .transpose()
            .map_err(environment_host_error)?;
        let permit = permit
            .as_ref()
            .map(|permit| permit as &dyn ExactHostAllocationPermit);
        let mut actions = ClaimedBacking::<BatchAction>::prepare(count, permit)?;
        let mut groups = ClaimedBacking::<ActivationGroup>::prepare(count, permit)?;
        let mut handles = ClaimedBacking::<GcHandle>::prepare(count, permit)?;
        let mut keys = ClaimedBacking::<(usize, BindingId, usize)>::prepare(count, permit)?;

        // Resolve inherited ownership before any visible mutation. The input
        // environment remains live while each distinct activation receives
        // its own current-operation GC root below.
        for change in changes {
            let (environment, binding, scope, source) = change.parts();
            let activation = resolve_activation(&environment, binding, scope)?;
            let next = source
                .as_ref()
                .map(WorkspaceSlotReadLease::shared_generation);
            handles.values.push(activation);
            actions.values.push(BatchAction {
                activation,
                group: 0,
                binding,
                next,
                retired: None,
                _source: source,
                _prior_roots: None,
            });
        }
        handles
            .values
            .sort_unstable_by_key(|handle| (handle.addr(), handle.epoch()));
        handles.values.dedup();

        for handle in &handles.values {
            let root = RootedValueLease::for_handle_in_context(*handle, context)
                .map_err(root_admission_error)?;
            let revision = gc_with_lexical_activation(handle, |activation| activation.revision())
                .map_err(environment_gc_error)?;
            groups.values.push(ActivationGroup {
                handle: *handle,
                _root: root,
                expected_revision: revision,
                next_revision: revision,
                missing: 0,
                prepared_table: None,
                table_receipt: None,
                retired_table: None,
            });
        }
        for index in 0..actions.values.len() {
            let action = &mut actions.values[index];
            let group = handles
                .values
                .binary_search_by_key(
                    &(action.activation.addr(), action.activation.epoch()),
                    |handle| (handle.addr(), handle.epoch()),
                )
                .expect("resolved lexical group");
            action.group = group;
            keys.values.push((group, action.binding, index));
            let revision = groups.values[group]
                .next_revision
                .checked_add(1)
                .ok_or_else(|| lexical_error("lexical revision exhausted"))?;
            groups.values[group].next_revision = revision;
        }

        // Validate unique targets in one in-place sort. The scratch Vec and
        // its claim are retired after preparation; action order stays source
        // ordered for deterministic old-owner release.
        keys.values.sort_unstable_by_key(|key| (key.0, key.1));
        for pair in keys.values.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
                return Err(lexical_error("duplicate lexical completion binding"));
            }
        }
        for action in &mut actions.values {
            let (exists, prior_roots) = gc_with_lexical_activation(
                &action.activation,
                |activation| -> Result<_, RuntimeError> {
                    let previous = activation.read(action.binding);
                    let roots = previous
                        .as_ref()
                        .map(|previous| {
                            RootedValueLease::for_value_in_context(previous.value(), context)
                                .map_err(root_admission_error)
                        })
                        .transpose()?;
                    Ok((activation.owns_binding(action.binding), roots))
                },
            )
            .map_err(environment_gc_error)??;
            action._prior_roots = prior_roots;
            if !exists {
                groups.values[action.group].missing += 1;
            }
        }
        for group_index in 0..groups.values.len() {
            let range = group_key_range(&keys.values, group_index);
            prepare_group_table(
                context,
                &mut groups.values[group_index],
                &actions.values,
                &keys.values[range],
            )?;
        }
        // The current collector traces the full reachable graph even during
        // minor collection, so the typed activation's new children are found
        // without a remembered-set publication. A future generation-pruned
        // collector must first add an exact admitted, post-checkpoint lexical
        // barrier owner; the existing allocating generic barrier cannot be
        // pre-recorded here because a major GC could clear it before commit.
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(environment_host_error)?;
        }
        Ok(Self {
            actions,
            groups,
            handles,
            keys,
            receipt,
        })
    }

    /// Acquire the short typed GC-exclusive gate and verify every activation
    /// revision before the paired Base workspace becomes visible.
    pub(crate) fn checkpoint<'a>(
        &'a mut self,
        context: &RuntimeContext,
    ) -> Result<ReadyLexicalCompletion<'a>, RuntimeError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(context)
                .map_err(environment_host_error)?;
        } else {
            HostAllocationReceipt::policy(Some(context)).map_err(environment_host_error)?;
        }
        for group in &self.groups.values {
            if let Some(receipt) = &group.table_receipt {
                receipt
                    .checkpoint(context)
                    .map_err(environment_host_error)?;
            }
        }
        let mut guard =
            gc_lock_lexical_mutations(&self.handles.values).map_err(environment_gc_error)?;
        for (index, group) in self.groups.values.iter().enumerate() {
            if guard.with_prevalidated_activation(index, |activation| activation.revision())
                != group.expected_revision
            {
                return Err(lexical_error(
                    "lexical activation changed during preparation",
                ));
            }
        }
        Ok(ReadyLexicalCompletion {
            actions: &mut self.actions,
            groups: &mut self.groups,
            keys: &self.keys.values,
            guard,
        })
    }
}

/// The short, already validated lexical commit half. No external callback or
/// fallible allocation is permitted while this owner exists.
pub(crate) struct ReadyLexicalCompletion<'a> {
    actions: &'a mut ClaimedBacking<BatchAction>,
    groups: &'a mut ClaimedBacking<ActivationGroup>,
    keys: &'a [(usize, BindingId, usize)],
    guard: LexicalMutationGuard<'a>,
}

impl<'a> ReadyLexicalCompletion<'a> {
    pub(crate) fn commit(mut self) -> LexicalCompletionRetirement<'a> {
        self.install_prevalidated();
        self.release_guard()
    }

    /// Install the source frame before releasing the lexical GC guard. The
    /// participant contract excludes callbacks and allocation in this span.
    pub(crate) fn commit_with_participant<P>(
        mut self,
        participant: &mut P,
    ) -> (LexicalCompletionRetirement<'a>, P::Retirement)
    where
        P: WorkspaceCompletionParticipant,
    {
        self.install_prevalidated();
        let retired_frame = participant.install_prevalidated();
        (self.release_guard(), retired_frame)
    }

    fn install_prevalidated(&mut self) {
        for (index, group) in self.groups.values.iter_mut().enumerate() {
            self.guard
                .with_prevalidated_activation(index, |activation| {
                    if let Some(prepared) = group.prepared_table.take() {
                        group.retired_table =
                            Some(activation.install_prepared_batch(prepared, group.next_revision));
                    } else {
                        for key in &self.keys[group_key_range(self.keys, index)] {
                            let action = &mut self.actions.values[key.2];
                            action.retired = activation
                                .upsert_prevalidated_batch(action.binding, action.next.take());
                        }
                        activation.set_prevalidated_revision(group.next_revision);
                    }
                });
        }
    }

    fn release_guard(self) -> LexicalCompletionRetirement<'a> {
        drop(self.guard);
        LexicalCompletionRetirement {
            actions: self.actions,
            groups: self.groups,
        }
    }
}

/// Old generations and backing retire only after both Base and GC mutable
/// guards are gone. Borrowing the prepared owner prevents premature refund.
pub(crate) struct LexicalCompletionRetirement<'a> {
    actions: &'a mut ClaimedBacking<BatchAction>,
    groups: &'a mut ClaimedBacking<ActivationGroup>,
}

impl Drop for LexicalCompletionRetirement<'_> {
    fn drop(&mut self) {
        for action in &mut self.actions.values {
            drop(action.retired.take());
        }
        for group in &mut self.groups.values {
            drop(group.retired_table.take());
        }
    }
}

fn resolve_activation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    scope: LexicalCompletionScope,
) -> Result<GcHandle, RuntimeError> {
    if scope == LexicalCompletionScope::Owned {
        return Ok(environment.handle());
    }
    // A function with inherited cells but no cells of its own reuses its
    // caller's activation. Only functions that own lexical bindings allocate
    // a child activation, so the inherited owner may be `environment` itself.
    let mut current = Some(environment.clone());
    while let Some(ancestor) = current {
        if gc_with_lexical_activation(&ancestor.handle(), |activation| {
            activation.owns_binding(binding)
        })
        .map_err(environment_gc_error)?
        {
            return Ok(ancestor.handle());
        }
        current = ancestor.parent();
    }
    Err(lexical_error(
        "inherited binding has no lexical activation owner",
    ))
}

#[cfg(test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn inherited_completion_uses_current_activation_when_no_child_was_created() {
    runmat_gc::gc_test_context(|| {
        let binding = BindingId(91);
        let (owner, _owner_roots) = super::environment::allocate_environment(None).unwrap();
        assert!(super::environment::reserve_owned_binding(&owner, binding).unwrap());
        assert_eq!(
            resolve_activation(&owner, binding, LexicalCompletionScope::Inherited).unwrap(),
            owner.handle()
        );

        let (child, _child_roots) =
            super::environment::allocate_environment(Some(owner.clone())).unwrap();
        assert_eq!(
            resolve_activation(&child, binding, LexicalCompletionScope::Inherited).unwrap(),
            owner.handle()
        );
    });
}

fn prepare_group_table(
    context: &RuntimeContext,
    group: &mut ActivationGroup,
    actions: &[BatchAction],
    keys: &[(usize, BindingId, usize)],
) -> Result<(), RuntimeError> {
    let (revision, len, capacity) = gc_with_lexical_activation(&group.handle, |activation| {
        (
            activation.revision(),
            activation.len(),
            activation.capacity(),
        )
    })
    .map_err(environment_gc_error)?;
    if revision != group.expected_revision {
        return Err(lexical_error(
            "lexical activation changed during preparation",
        ));
    }
    if group.missing <= capacity - len {
        return Ok(());
    }
    let required = len
        .checked_add(group.missing)
        .ok_or_else(|| lexical_error("lexical table capacity overflow"))?;
    let grown = capacity.checked_mul(2).unwrap_or(required).max(1);
    let target = required.max(grown);
    let raw = PreparedLexicalTable::requested_bytes(target)
        .map_err(|error| lexical_error(&error.to_string()))?;
    let raw = u64::try_from(raw).map_err(|_| lexical_error("lexical table size overflow"))?;
    let quote =
        HostAllocationPlan::retained_exact_output_bytes(raw, 1).map_err(environment_host_error)?;
    let receipt = HostAllocationReceipt::admit(
        Some(context),
        HostAllocationPlan::new(0, quote).map_err(environment_host_error)?,
    )
    .map_err(environment_host_error)?;
    let permit = receipt
        .as_ref()
        .map(|receipt| receipt.output_permit(context))
        .transpose()
        .map_err(environment_host_error)?;
    let mut prepared = PreparedLexicalTable::prepare(
        target,
        revision,
        permit
            .as_ref()
            .map(|permit| permit as &dyn ExactHostAllocationPermit),
    )
    .map_err(|error| lexical_error(&error.to_string()))?;
    let copied = gc_with_lexical_activation(&group.handle, |activation| {
        prepared.populate_from(activation)
    })
    .map_err(environment_gc_error)?;
    if !copied {
        return Err(lexical_error(
            "lexical activation changed during table preparation",
        ));
    }
    for key in keys {
        let action = &actions[key.2];
        prepared
            .apply(action.binding, action.next.clone())
            .map_err(|unused| {
                drop(unused);
                lexical_error("prepared lexical table capacity was exceeded")
            })?;
    }
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(context)
            .map_err(environment_host_error)?;
    }
    group.prepared_table = Some(prepared);
    group.table_receipt = receipt;
    Ok(())
}

fn group_key_range(keys: &[(usize, BindingId, usize)], group: usize) -> std::ops::Range<usize> {
    let start = keys.partition_point(|key| key.0 < group);
    let end = keys.partition_point(|key| key.0 <= group);
    start..end
}

fn lexical_error(message: &str) -> RuntimeError {
    semantic_error("LexicalEnvironment", message)
}
