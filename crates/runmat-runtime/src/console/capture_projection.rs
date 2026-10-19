//! Poll-local capture stack and entry owners.

mod frame;

use self::frame::CaptureFrameOwner;
use super::{CapturedConsoleEntries, ConsoleEntry};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    HostOutputPermit,
};
use crate::context::overlay_storage::AdmittedOverlayVec;
use crate::context::poll_projection::PollProjection;
use crate::context::RuntimeContext;
use runmat_value::exact_capacity_vec;

pub(crate) enum CaptureRoute {
    Captured,
    Uncaptured(ConsoleEntry),
}

pub(crate) struct ConsoleCaptureProjection {
    next_token: u64,
    inherited: Vec<CaptureFrameOwner>,
    frames: AdmittedOverlayVec<(u64, CaptureFrameOwner)>,
}

impl Default for ConsoleCaptureProjection {
    fn default() -> Self {
        Self {
            next_token: 0,
            inherited: Vec::new(),
            frames: AdmittedOverlayVec::unbounded(),
        }
    }
}

impl ConsoleCaptureProjection {
    fn ensure_stack_policy(
        owner: &PollProjection<Self>,
        context: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        if HostAllocationReceipt::policy(Some(context))? != HostAllocationPolicy::Managed {
            return Ok(());
        }
        let needs_managed = matches!(&owner.borrow().frames, AdmittedOverlayVec::Unbounded { .. });
        if !needs_managed {
            return Ok(());
        }
        if !owner.borrow().frames.is_empty() {
            return Err(HostAllocationError::MissingManagedOwner);
        }
        let receipt = HostAllocationReceipt::admit(Some(context), HostAllocationPlan::new(0, 0)?)?
            .ok_or(HostAllocationError::MissingManagedOwner)?;
        let managed = AdmittedOverlayVec::managed(context, &receipt)?;
        receipt.checkpoint(context)?;
        let mut managed = Some(managed);
        let retired = {
            let mut projection = owner.borrow_mut();
            if matches!(&projection.frames, AdmittedOverlayVec::Unbounded { .. })
                && projection.frames.is_empty()
            {
                Some(std::mem::replace(
                    &mut projection.frames,
                    managed.take().expect("prepared stack"),
                ))
            } else {
                None
            }
        };
        drop(managed);
        drop(retired);
        Ok(())
    }

    fn active_frames(&self) -> impl Iterator<Item = &CaptureFrameOwner> {
        self.inherited
            .iter()
            .chain(self.frames.as_slice().iter().map(|(_, frame)| frame))
            .filter(|frame| frame.active())
    }

    pub(crate) fn poll_snapshot_scratch_bytes(&self) -> Option<u64> {
        self.active_frames()
            .count()
            .checked_mul(std::mem::size_of::<CaptureFrameOwner>())
            .and_then(|bytes| u64::try_from(bytes).ok())
    }

    pub(crate) fn snapshot(&self) -> Self {
        Self {
            next_token: 0,
            inherited: self.active_frames().cloned().collect(),
            frames: AdmittedOverlayVec::unbounded(),
        }
    }

    pub(crate) fn poll_snapshot(
        &self,
        context: &RuntimeContext,
        permit: HostOutputPermit,
    ) -> Result<Self, HostAllocationError> {
        let mut inherited = exact_capacity_vec(self.active_frames().count())
            .map_err(HostAllocationError::Numeric)?;
        inherited.extend(self.active_frames().cloned());
        let frames = AdmittedOverlayVec::managed_with_permit(context, permit)?;
        Ok(Self {
            next_token: 0,
            inherited,
            frames,
        })
    }

    pub(crate) fn record(
        owner: &PollProjection<Self>,
        entry: ConsoleEntry,
        context: &RuntimeContext,
    ) -> Result<CaptureRoute, HostAllocationError> {
        let frame = owner.borrow().active_frames().last().cloned();
        let Some(frame) = frame else {
            return Ok(CaptureRoute::Uncaptured(entry));
        };
        frame.record(entry, context)?;
        Ok(CaptureRoute::Captured)
    }

    pub(crate) fn begin(
        owner: &PollProjection<Self>,
        context: &RuntimeContext,
    ) -> Result<CaptureLease, HostAllocationError> {
        Self::ensure_stack_policy(owner, context)?;
        let (token, next, snapshot) = {
            let projection = owner.borrow();
            let token = projection.next_token;
            (
                token,
                token.checked_add(1).ok_or(HostAllocationError::Overflow)?,
                projection.frames.snapshot_push()?,
            )
        };
        let frame = CaptureFrameOwner::prepare(context)?;
        let prepared =
            AdmittedOverlayVec::<(u64, CaptureFrameOwner)>::prepare_push(context, snapshot)?;
        let prepared = prepared.checkpoint(context).map_err(|(error, _)| error)?;
        let result = {
            let mut projection = owner.borrow_mut();
            let result = projection
                .frames
                .install_push(prepared, (token, frame.clone()));
            if result.is_ok() {
                projection.next_token = next;
            }
            result
        };
        match result {
            Ok(retirement) => {
                drop(retirement);
                Ok(CaptureLease {
                    owner: owner.clone(),
                    token,
                    frame,
                })
            }
            Err(failure) => {
                let error = failure.cause;
                drop(failure);
                Err(error)
            }
        }
    }
}

pub(crate) struct CaptureLease {
    owner: PollProjection<ConsoleCaptureProjection>,
    token: u64,
    frame: CaptureFrameOwner,
}

impl CaptureLease {
    pub(crate) fn finish(self) -> CapturedConsoleEntries {
        let entries = self.frame.take_entries();
        drop(self);
        CapturedConsoleEntries::from_storage(entries)
    }
}

impl Drop for CaptureLease {
    fn drop(&mut self) {
        self.frame.deactivate();
        let retired = {
            let mut projection = self.owner.borrow_mut();
            let index = projection
                .frames
                .as_slice()
                .iter()
                .position(|(token, frame)| *token == self.token && frame.ptr_eq(&self.frame));
            index.and_then(|index| projection.frames.retire(index))
        };
        drop(retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleStream;

    fn entry(text: &str) -> ConsoleEntry {
        ConsoleEntry {
            stream: ConsoleStream::Stdout,
            text: text.into(),
            timestamp_ms: 0,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn inherited_outer_capture_survives_child_overlay_but_closure_is_exact() {
        let context = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let parent = PollProjection::legacy(ConsoleCaptureProjection::default());
        let outer = ConsoleCaptureProjection::begin(&parent, &context).unwrap();
        let child = PollProjection::legacy(parent.borrow().snapshot());
        let inner = ConsoleCaptureProjection::begin(&child, &context).unwrap();
        assert!(matches!(
            ConsoleCaptureProjection::record(&child, entry("inner"), &context).unwrap(),
            CaptureRoute::Captured
        ));
        assert_eq!(inner.finish()[0].text, "inner");
        assert!(matches!(
            ConsoleCaptureProjection::record(&child, entry("outer"), &context).unwrap(),
            CaptureRoute::Captured
        ));
        assert_eq!(outer.finish()[0].text, "outer");
        let CaptureRoute::Uncaptured(entry) =
            ConsoleCaptureProjection::record(&child, entry("after parent retirement"), &context)
                .unwrap()
        else {
            panic!("retired capture must return the entry")
        };
        assert_eq!(entry.text, "after parent retirement");
    }
}
