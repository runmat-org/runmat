//! Claimed, closeable authority for one session's compiler binding table.
//!
//! An observer may outlive its session, but it retains only this small exact
//! control after `close` retires the table. A move-only phase lease excludes
//! preparation reentry without holding a table borrow across domain admission
//! callbacks. Physical table growth uses the sealed Runtime allocator under
//! a short table borrow.

use super::binding_table::BindingTable;
use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::{RuntimeContext, RuntimeContextLocalId};
use runmat_value::{ExactHostAllocationClaim, ExactHostNumericAllocationError};
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::{fmt, ops::Deref};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::session) enum BindingPhase {
    Ingress,
    Import,
    Publication,
    SlotRefresh,
    Clear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhaseState {
    Idle,
    Held {
        phase: BindingPhase,
        checkpointed: bool,
    },
}

struct BindingControlInner {
    table: RefCell<Option<BindingTable>>,
    phase: Cell<PhaseState>,
    closed: Cell<bool>,
    revision: Cell<u64>,
    runtime: RuntimeContextLocalId,
    // The ExactShared header is deallocated before this payload and claim.
    _header_claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone)]
pub(in crate::session) struct BindingControl(ExactShared<BindingControlInner>);

pub(in crate::session) struct BindingReadGuard<'a> {
    value: Option<Ref<'a, BindingTable>>,
    inner: &'a BindingControlInner,
}

impl Deref for BindingReadGuard<'_> {
    type Target = BindingTable;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().expect("binding read guard is live")
    }
}

impl Drop for BindingReadGuard<'_> {
    fn drop(&mut self) {
        self.value.take();
        if self.inner.closed.get() && self.inner.phase.get() == PhaseState::Idle {
            BindingControl::retire_closed_table(self.inner);
        }
    }
}

impl fmt::Debug for BindingControl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BindingControl")
    }
}

pub(in crate::session) struct BindingLease {
    control: BindingControl,
    phase: BindingPhase,
    next_revision: u64,
    active: bool,
}

impl BindingControl {
    pub(in crate::session) fn new_for_context(
        context: &RuntimeContext,
        table: BindingTable,
    ) -> Result<Self, HostAllocationError> {
        let bytes = ExactShared::<BindingControlInner>::requested_bytes();
        let quoted = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?;
        let plan = HostAllocationPlan::new(0, quoted)?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)?;
        let claim = match receipt.as_ref() {
            Some(receipt) => Some(
                receipt
                    .with_output_allocation(context, |permit| permit.reserve_exact_claim(bytes))?,
            ),
            None => None,
        };
        let inner = BindingControlInner {
            table: RefCell::new(Some(table)),
            phase: Cell::new(PhaseState::Idle),
            closed: Cell::new(false),
            revision: Cell::new(0),
            runtime: context.local_identity(),
            _header_claim: claim,
        };
        let control = ExactShared::try_new(inner).map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        if let Some(receipt) = receipt.as_ref() {
            receipt.checkpoint(context)?;
        }
        Ok(Self(control))
    }

    fn inner(&self) -> &BindingControlInner {
        self.0.get()
    }

    pub(super) fn reserve(
        &self,
        phase: BindingPhase,
        context: Option<&RuntimeContext>,
    ) -> Result<BindingLease, HostAllocationError> {
        let inner = self.inner();
        if inner.closed.get()
            || inner.phase.get() != PhaseState::Idle
            || context.is_some_and(|context| inner.runtime != context.local_identity())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        // Existing readers must retire before a phase begins. Reads during
        // preparation remain possible; checkpoint excludes them before the
        // canonical effect and then fences all new public readers.
        drop(
            inner
                .table
                .try_borrow_mut()
                .map_err(|_| HostAllocationError::StaleAuthority)?,
        );
        let next_revision = inner
            .revision
            .get()
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        inner.phase.set(PhaseState::Held {
            phase,
            checkpointed: false,
        });
        Ok(BindingLease {
            control: self.clone(),
            phase,
            next_revision,
            active: true,
        })
    }

    pub(super) fn try_borrow(&self) -> Result<BindingReadGuard<'_>, ()> {
        if self.inner().closed.get()
            || matches!(
                self.inner().phase.get(),
                PhaseState::Held {
                    checkpointed: true,
                    ..
                }
            )
        {
            return Err(());
        }
        let table = self.inner().table.try_borrow().map_err(|_| ())?;
        let value = Ref::filter_map(table, Option::as_ref).map_err(|_| ())?;
        Ok(BindingReadGuard {
            value: Some(value),
            inner: self.inner(),
        })
    }

    pub(in crate::session) fn borrow(&self) -> BindingReadGuard<'_> {
        self.try_borrow().expect("session binding table is live")
    }

    #[cfg(test)]
    pub(super) fn borrow_mut(&self) -> RefMut<'_, BindingTable> {
        assert_eq!(self.inner().phase.get(), PhaseState::Idle);
        let table = self.inner().table.borrow_mut();
        RefMut::map(table, |table| {
            table.as_mut().expect("session binding table is live")
        })
    }

    pub(in crate::session) fn close(&self) {
        let inner = self.inner();
        inner.closed.set(true);
        if inner.phase.get() == PhaseState::Idle {
            Self::retire_closed_table(inner);
        }
    }

    fn retire_closed_table(inner: &BindingControlInner) {
        if let Ok(mut table) = inner.table.try_borrow_mut() {
            table.take();
        }
    }
}

impl BindingLease {
    fn valid(&self) -> bool {
        self.active
            && !self.control.inner().closed.get()
            && matches!(
                self.control.inner().phase.get(),
                PhaseState::Held { phase, .. } if phase == self.phase
            )
    }

    pub(super) fn checkpoint(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<(), HostAllocationError> {
        if !self.valid()
            || context
                .is_some_and(|context| self.control.inner().runtime != context.local_identity())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        drop(
            self.control
                .inner()
                .table
                .try_borrow_mut()
                .map_err(|_| HostAllocationError::StaleAuthority)?,
        );
        self.control.inner().phase.set(PhaseState::Held {
            phase: self.phase,
            checkpointed: true,
        });
        Ok(())
    }

    /// Cancel a final publication fence without relinquishing the pending
    /// participant's mutation lease. No table update has happened yet.
    pub(super) fn rollback_checkpoint(&self) {
        if self.valid() {
            self.control.inner().phase.set(PhaseState::Held {
                phase: self.phase,
                checkpointed: false,
            });
        }
    }

    pub(super) fn try_borrow(&self) -> Result<Ref<'_, BindingTable>, ()> {
        if !self.valid() {
            return Err(());
        }
        let table = self.control.inner().table.try_borrow().map_err(|_| ())?;
        Ref::filter_map(table, Option::as_ref).map_err(|_| ())
    }

    pub(super) fn try_borrow_mut(&self) -> Result<RefMut<'_, BindingTable>, ()> {
        if !self.valid() {
            return Err(());
        }
        let table = self
            .control
            .inner()
            .table
            .try_borrow_mut()
            .map_err(|_| ())?;
        RefMut::filter_map(table, Option::as_mut).map_err(|_| ())
    }

    pub(super) fn replace_table(&self, table: BindingTable) -> Result<(), BindingTable> {
        let Ok(mut installed) = self.try_borrow_mut() else {
            return Err(table);
        };
        *installed = table;
        self.control.inner().revision.set(self.next_revision);
        Ok(())
    }

    pub(super) fn mark_changed(&self) {
        assert!(self.valid(), "binding mutation requires a live phase lease");
        self.control.inner().revision.set(self.next_revision);
    }
}

impl Drop for BindingLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let inner = self.control.inner();
        if matches!(inner.phase.get(), PhaseState::Held { phase, .. } if phase == self.phase) {
            inner.phase.set(PhaseState::Idle);
        }
        if inner.closed.get() {
            BindingControl::retire_closed_table(inner);
        }
        self.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_runtime::execution::RuntimeExecutionService;
    use std::rc::Rc;

    #[test]
    fn phase_checkpoint_fences_readers_and_close_revokes_retained_observer() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let table = BindingTable::new_for_context(&context).unwrap();
        let control = BindingControl::new_for_context(&context, table).unwrap();
        let observer = control.clone();
        let reader = control.borrow();
        assert!(control
            .reserve(BindingPhase::Publication, Some(&context))
            .is_err());
        drop(reader);
        let lease = control
            .reserve(BindingPhase::Publication, Some(&context))
            .unwrap();
        let preparation_reader = observer.try_borrow().unwrap();
        assert!(lease.checkpoint(Some(&context)).is_err());
        drop(preparation_reader);
        lease.checkpoint(Some(&context)).unwrap();
        assert!(observer.try_borrow().is_err());
        control.close();
        assert!(lease.checkpoint(Some(&context)).is_err());
        drop(lease);
        assert!(observer.try_borrow().is_err());
    }
}
