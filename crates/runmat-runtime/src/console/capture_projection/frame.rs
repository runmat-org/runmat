//! Exact shared frame header and its separately admitted entry collection.

use super::super::ConsoleEntry;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::overlay_storage::AdmittedOverlayVec;
use crate::context::RuntimeContext;
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

struct CaptureFrame {
    active: Cell<bool>,
    entries: RefCell<AdmittedOverlayVec<ConsoleEntry>>,
    _header_claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone)]
enum FrameBacking {
    Unbounded(Rc<CaptureFrame>),
    Managed(ExactShared<CaptureFrame>),
}

#[derive(Clone)]
pub(super) struct CaptureFrameOwner(FrameBacking);

impl CaptureFrameOwner {
    pub(super) fn prepare(context: &RuntimeContext) -> Result<Self, HostAllocationError> {
        match HostAllocationReceipt::policy(Some(context))? {
            HostAllocationPolicy::Unbounded => {
                Ok(Self(FrameBacking::Unbounded(Rc::new(CaptureFrame {
                    active: Cell::new(true),
                    entries: RefCell::new(AdmittedOverlayVec::unbounded()),
                    _header_claim: None,
                }))))
            }
            HostAllocationPolicy::Managed => {
                let header = u64::try_from(ExactShared::<CaptureFrame>::requested_bytes())
                    .map_err(|_| HostAllocationError::Overflow)?;
                let retained = HostAllocationPlan::retained_exact_output_bytes(header, 1)?;
                let receipt = HostAllocationReceipt::admit(
                    Some(context),
                    HostAllocationPlan::new(0, retained)?,
                )?
                .ok_or(HostAllocationError::MissingManagedOwner)?;
                let permit = receipt.output_permit(context)?;
                let claim =
                    permit.reserve_exact_claim(ExactShared::<CaptureFrame>::requested_bytes());
                receipt.checkpoint(context)?;
                let claim = claim.map_err(HostAllocationError::Numeric)?;
                let entries = AdmittedOverlayVec::managed_with_permit(context, permit)?;
                let frame = CaptureFrame {
                    active: Cell::new(true),
                    entries: RefCell::new(entries),
                    _header_claim: Some(claim),
                };
                let owner = ExactShared::try_new(frame);
                receipt.checkpoint(context)?;
                let owner = owner.map_err(|_| {
                    HostAllocationError::Numeric(
                        runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                    )
                })?;
                Ok(Self(FrameBacking::Managed(owner)))
            }
        }
    }

    fn frame(&self) -> &CaptureFrame {
        match &self.0 {
            FrameBacking::Unbounded(frame) => frame,
            FrameBacking::Managed(frame) => frame.get(),
        }
    }

    pub(super) fn active(&self) -> bool {
        self.frame().active.get()
    }
    pub(super) fn deactivate(&self) {
        self.frame().active.set(false);
    }
    pub(super) fn entries(&self) -> &RefCell<AdmittedOverlayVec<ConsoleEntry>> {
        &self.frame().entries
    }
    pub(super) fn record(
        &self,
        entry: ConsoleEntry,
        context: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        if !self.active() {
            return Err(HostAllocationError::StaleAuthority);
        }
        let snapshot = self.entries().borrow().snapshot_push()?;
        let prepared = AdmittedOverlayVec::<ConsoleEntry>::prepare_push(context, snapshot)?;
        let prepared = prepared.checkpoint(context).map_err(|(error, _)| error)?;
        if !self.active() {
            return Err(HostAllocationError::StaleAuthority);
        }
        let result = self.entries().borrow_mut().install_push(prepared, entry);
        match result {
            Ok(retirement) => {
                drop(retirement);
                Ok(())
            }
            Err(failure) => {
                let error = failure.cause;
                drop(failure);
                Err(error)
            }
        }
    }
    pub(super) fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (FrameBacking::Unbounded(left), FrameBacking::Unbounded(right)) => {
                Rc::ptr_eq(left, right)
            }
            (FrameBacking::Managed(left), FrameBacking::Managed(right)) => left.ptr_eq(right),
            _ => false,
        }
    }
    pub(super) fn take_entries(&self) -> AdmittedOverlayVec<ConsoleEntry> {
        std::mem::replace(
            &mut *self.entries().borrow_mut(),
            AdmittedOverlayVec::unbounded(),
        )
    }
}
