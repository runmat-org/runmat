//! Persistent compiler-binding participant for staged workspace effects.

#[path = "pending/validation.rs"]
mod validation;

use runmat_runtime::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use runmat_runtime::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_runtime::workspace::WorkspacePublicationStamp;

use super::super::binding_control::{BindingControl, BindingLease, BindingPhase};
use super::patch::{PendingBindingPatch, PreparedPatchUpdate};
use crate::session::SessionWorkspaceBinding;
use runmat_runtime::workspace::WorkspaceName;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::session::workspace::publication) struct PendingUpdateToken(u64);

struct StagedUpdate {
    token: PendingUpdateToken,
    update: StagedPatch,
    stamp: WorkspacePublicationStamp,
}

enum StagedPatch {
    Incremental(PreparedPatchUpdate),
    Replacement(PendingBindingPatch),
}

/// Private participant state. The retained lease is deliberately not
/// checkpointed while language execution can suspend: public binding reads
/// remain available, but ingress/import/slot-refresh cannot race this patch.
pub(in crate::session::workspace::publication) struct PendingBindingParticipant {
    lease: BindingLease,
    patch: Option<PendingBindingPatch>,
    runtime: RuntimeContextLocalId,
    anchor: WorkspacePublicationStamp,
    latest: Option<WorkspacePublicationStamp>,
    next_token: u64,
    staged: Option<StagedUpdate>,
    final_checkpointed: bool,
}

/// Binding phase reserved before admission/allocator callbacks are entered.
/// This closes the quotation-to-install race without fencing ordinary reads.
pub(in crate::session::workspace::publication) struct PendingBindingStart {
    lease: BindingLease,
    runtime: RuntimeContextLocalId,
    anchor: WorkspacePublicationStamp,
}

impl PendingBindingStart {
    pub(in crate::session::workspace::publication) fn reserve(
        bindings: &BindingControl,
        context: &RuntimeContext,
        stamp: WorkspacePublicationStamp,
    ) -> Result<Self, HostAllocationError> {
        Ok(Self {
            lease: bindings.reserve(BindingPhase::Publication, Some(context))?,
            runtime: context.local_identity(),
            anchor: stamp,
        })
    }

    pub(in crate::session::workspace::publication) fn table_growth_retained_bytes(
        &self,
        possible_insertions: usize,
    ) -> Result<u64, HostAllocationError> {
        self.lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?
            .growth_retained_bytes(possible_insertions)
    }

    pub(in crate::session::workspace::publication) fn with_installed_binding<T>(
        &self,
        name: &str,
        inspect: impl FnOnce(Option<(&WorkspaceName, &SessionWorkspaceBinding)>) -> T,
    ) -> Result<T, HostAllocationError> {
        let table = self
            .lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?;
        Ok(inspect(table.get_key_value(name)))
    }

    pub(in crate::session::workspace::publication) fn finish(
        self,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<PendingBindingParticipant, HostAllocationError> {
        if self.runtime != context.local_identity() {
            return Err(HostAllocationError::StaleAuthority);
        }
        let patch = PendingBindingPatch::prepare_empty(Some(context), receipt)?;
        Ok(PendingBindingParticipant {
            lease: self.lease,
            patch: Some(patch),
            runtime: self.runtime,
            anchor: self.anchor,
            latest: None,
            next_token: 0,
            staged: None,
            final_checkpointed: false,
        })
    }
}

impl PendingBindingParticipant {
    pub(in crate::session::workspace::publication) fn owner_retained_bytes(
    ) -> Result<u64, HostAllocationError> {
        PendingBindingPatch::owner_retained_bytes()
    }

    pub(in crate::session::workspace::publication) fn first_action_retained_bytes(
    ) -> Result<u64, HostAllocationError> {
        PendingBindingPatch::first_action_retained_bytes()
    }

    pub(in crate::session::workspace::publication) fn matches(
        &self,
        context: &RuntimeContext,
        stamp: WorkspacePublicationStamp,
    ) -> bool {
        !self.final_checkpointed
            && self.runtime == context.local_identity()
            && self
                .latest
                .map_or(stamp == self.anchor, |previous| stamp.continues(previous))
            && self.staged.is_none()
    }

    pub(in crate::session::workspace::publication) fn matches_completion(
        &self,
        context: &RuntimeContext,
        stamp: WorkspacePublicationStamp,
    ) -> bool {
        !self.final_checkpointed
            && self.runtime == context.local_identity()
            && self
                .latest
                .map_or(stamp == self.anchor, |latest| stamp.continues(latest))
            && self.staged.is_none()
    }

    pub(in crate::session::workspace::publication) fn patch(&self) -> &PendingBindingPatch {
        self.patch.as_ref().expect("pending patch is live")
    }

    pub(in crate::session::workspace::publication) fn with_installed_binding<T>(
        &self,
        name: &str,
        inspect: impl FnOnce(Option<(&WorkspaceName, &SessionWorkspaceBinding)>) -> T,
    ) -> Result<T, HostAllocationError> {
        let table = self
            .lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?;
        Ok(inspect(table.get_key_value(name)))
    }

    pub(in crate::session::workspace::publication) fn patch_growth_retained_bytes(
        &self,
        name: &str,
    ) -> Result<u64, HostAllocationError> {
        self.patch().growth_retained_bytes(name)
    }

    /// Reserve physical collection growth and install a private update token.
    /// This returns only after the operation receipt is checkpointed; no
    /// RuntimeContext or output permit is retained across canonical install.
    pub(in crate::session::workspace::publication) fn prepare_update(
        &mut self,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        update: PreparedPatchUpdate,
        stamp: WorkspacePublicationStamp,
    ) -> Result<PendingUpdateToken, HostAllocationError> {
        if !self.matches(context, stamp) {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.validate_update_slots(&update)?;
        let insertions = self.patch().insertions_after(&update)?;
        if let Some(name) = PendingBindingPatch::update_name(&update) {
            self.patch
                .as_mut()
                .expect("pending patch is live")
                .reserve_action(Some(context), receipt, name)?;
        }
        match receipt {
            Some(receipt) => self
                .lease
                .try_borrow_mut()
                .map_err(|_| HostAllocationError::StaleAuthority)?
                .reserve_admitted(context, Some(receipt), insertions)?,
            None => self
                .lease
                .try_borrow_mut()
                .map_err(|_| HostAllocationError::StaleAuthority)?
                .reserve_unbounded(insertions)?,
        }
        if let Some(receipt) = receipt {
            receipt.checkpoint(context)?;
        }
        let generation = self
            .next_token
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let token = PendingUpdateToken(generation);
        self.next_token = generation;
        self.staged = Some(StagedUpdate {
            token,
            update: StagedPatch::Incremental(update),
            stamp,
        });
        Ok(token)
    }

    /// Install one already coalesced whole-completion replacement. The caller
    /// prepared every deep name/key owner while constructing `replacement`.
    pub(in crate::session::workspace::publication) fn prepare_replacement(
        &mut self,
        context: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
        replacement: PendingBindingPatch,
        stamp: WorkspacePublicationStamp,
    ) -> Result<PendingUpdateToken, HostAllocationError> {
        if !self.matches_completion(context, stamp) {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.validate_replacement_slots(&replacement)?;
        let insertions = replacement.possible_insertions();
        match receipt {
            Some(receipt) => self
                .lease
                .try_borrow_mut()
                .map_err(|_| HostAllocationError::StaleAuthority)?
                .reserve_admitted(context, Some(receipt), insertions)?,
            None => self
                .lease
                .try_borrow_mut()
                .map_err(|_| HostAllocationError::StaleAuthority)?
                .reserve_unbounded(insertions)?,
        }
        if let Some(receipt) = receipt {
            receipt.checkpoint(context)?;
        }
        let generation = self
            .next_token
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let token = PendingUpdateToken(generation);
        self.next_token = generation;
        self.staged = Some(StagedUpdate {
            token,
            update: StagedPatch::Replacement(replacement),
            stamp,
        });
        Ok(token)
    }

    /// Called only after Runtime installs the corresponding prepared mutation.
    pub(in crate::session::workspace::publication) fn commit_update(
        &mut self,
        token: PendingUpdateToken,
    ) {
        assert!(
            self.staged
                .as_ref()
                .is_some_and(|staged| staged.token == token),
            "pending binding update token must match its preparation"
        );
        let staged = self.staged.take().expect("matching update exists");
        match staged.update {
            StagedPatch::Incremental(update) => self
                .patch
                .as_mut()
                .expect("pending patch is live")
                .apply_update(update),
            StagedPatch::Replacement(replacement) => {
                let retired = self.patch.replace(replacement);
                drop(retired);
            }
        }
        self.latest = Some(staged.stamp);
    }

    pub(in crate::session::workspace::publication) fn abort_update(
        &mut self,
        token: PendingUpdateToken,
    ) {
        if self
            .staged
            .as_ref()
            .is_some_and(|staged| staged.token == token)
        {
            self.staged.take();
        }
    }

    /// Fence readers immediately before canonical publication. Callers must
    /// not await after this succeeds.
    pub(in crate::session::workspace::publication) fn checkpoint_final(
        &mut self,
        context: &RuntimeContext,
        final_stamp: WorkspacePublicationStamp,
    ) -> Result<(), HostAllocationError> {
        if self.runtime != context.local_identity()
            || self.staged.is_some()
            || self.final_checkpointed
            || self
                .latest
                .is_none_or(|latest| !final_stamp.same_destination(latest))
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.validate_replacement_slots(self.patch())?;
        self.lease.checkpoint(Some(context))?;
        self.final_checkpointed = true;
        Ok(())
    }

    /// Fence a whole completion while its replacement remains private. The
    /// caller may still abort without losing an earlier pending prefix.
    pub(in crate::session::workspace::publication) fn checkpoint_completion_final(
        &mut self,
        context: &RuntimeContext,
        token: PendingUpdateToken,
        final_stamp: WorkspacePublicationStamp,
    ) -> Result<(), HostAllocationError> {
        if self.runtime != context.local_identity() || self.final_checkpointed {
            return Err(HostAllocationError::StaleAuthority);
        }
        let staged = self
            .staged
            .as_ref()
            .ok_or(HostAllocationError::StaleAuthority)?;
        if staged.token != token || !final_stamp.same_destination(staged.stamp) {
            return Err(HostAllocationError::StaleAuthority);
        }
        let StagedPatch::Replacement(replacement) = &staged.update else {
            return Err(HostAllocationError::StaleAuthority);
        };
        self.validate_replacement_slots(replacement)?;
        self.lease.checkpoint(Some(context))?;
        self.final_checkpointed = true;
        Ok(())
    }

    /// Final refusal leaves the admitted prefix available for a retry.
    pub(in crate::session::workspace::publication) fn abort_final(&mut self) {
        if self.final_checkpointed {
            self.lease.rollback_checkpoint();
            self.final_checkpointed = false;
        }
    }

    /// Apply after the workspace commit. All names, keys, slots and table
    /// capacity were prepared by earlier update operations.
    pub(in crate::session::workspace::publication) fn commit_final(mut self) {
        assert!(
            self.final_checkpointed,
            "pending binding patch was not fenced"
        );
        let patch = self.patch.take().expect("pending patch is live");
        let mut table = self
            .lease
            .try_borrow_mut()
            .expect("checkpointed binding participant retains its table");
        patch.apply_to_table(&mut table);
        drop(table);
        self.lease.mark_changed();
    }
}

impl Drop for PendingBindingParticipant {
    fn drop(&mut self) {
        // Retire independently claimed prepared owners before the phase lease
        // exposes the compiler table to another mutator.
        self.staged.take();
        self.patch.take();
    }
}
