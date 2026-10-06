//! Exact physical String blocks under one already-admitted diagnostic claim.
//! This owner charges bytes, not a whole error or complete invocation.

use super::{HostEffectAllocationError, HostEffectLiveAllocationClaim};
use std::alloc::{alloc, dealloc, Layout};
use std::fmt::{self, Write as _};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

pub(crate) struct ExactHostEffectDiagnosticStrings {
    remaining: usize,
    claim: Option<HostEffectLiveAllocationClaim>,
}

impl ExactHostEffectDiagnosticStrings {
    /// `authorized_bytes` must be exactly the sum of physical String payload
    /// capacities reserved by `claim` at the owning stage/receipt boundary.
    pub(crate) fn new(authorized_bytes: usize, claim: HostEffectLiveAllocationClaim) -> Self {
        Self {
            remaining: authorized_bytes,
            claim: Some(claim),
        }
    }

    pub(crate) fn copy(&mut self, text: &str) -> Result<String, HostEffectAllocationError> {
        self.allocate(text.len(), |output| output.write_str(text))
    }

    pub(crate) fn render(
        &mut self,
        requested_bytes: usize,
        text: impl fmt::Display,
    ) -> Result<String, HostEffectAllocationError> {
        self.allocate(requested_bytes, |output| write!(output, "{text}"))
    }

    fn allocate(
        &mut self,
        requested_bytes: usize,
        write_text: impl FnOnce(&mut ExactStringWriter) -> fmt::Result,
    ) -> Result<String, HostEffectAllocationError> {
        if requested_bytes > self.remaining {
            return Err(HostEffectAllocationError::ExceedsStageEnvelope);
        }
        let mut output = ExactStringWriter::new(requested_bytes)?;
        write_text(&mut output).map_err(|_| HostEffectAllocationError::InvalidInvocation)?;
        let text = output.finish()?;
        self.remaining -= requested_bytes;
        Ok(text)
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        if self.remaining != 0 {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.claim
            .take()
            .ok_or(HostEffectAllocationError::InvalidInvocation)
    }
}

struct ExactStringWriter {
    pointer: NonNull<u8>,
    capacity: usize,
    length: usize,
}

impl ExactStringWriter {
    fn new(capacity: usize) -> Result<Self, HostEffectAllocationError> {
        let layout =
            Layout::array::<u8>(capacity).map_err(|_| HostEffectAllocationError::Overflow)?;
        let pointer = if capacity == 0 {
            NonNull::dangling()
        } else {
            // SAFETY: the checked layout is retained for matching deallocation
            // on failure or transferred unchanged to String on success.
            NonNull::new(unsafe { alloc(layout) })
                .ok_or(HostEffectAllocationError::AllocationFailed)?
        };
        Ok(Self {
            pointer,
            capacity,
            length: 0,
        })
    }

    fn finish(self) -> Result<String, HostEffectAllocationError> {
        if self.length != self.capacity {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let output = ManuallyDrop::new(self);
        if output.capacity == 0 {
            return Ok(String::new());
        }
        // SAFETY: every appended segment was valid UTF-8, the initialized
        // length equals capacity, and the global allocator supplied exactly
        // this layout. String takes the unique pointer and deallocates it.
        Ok(unsafe {
            String::from_raw_parts(output.pointer.as_ptr(), output.length, output.capacity)
        })
    }
}

impl fmt::Write for ExactStringWriter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let next = self.length.checked_add(text.len()).ok_or(fmt::Error)?;
        if next > self.capacity {
            return Err(fmt::Error);
        }
        if !text.is_empty() {
            // SAFETY: the checked range is within the unique allocation and
            // does not overlap the borrowed text segment.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    text.as_ptr(),
                    self.pointer.as_ptr().add(self.length),
                    text.len(),
                );
            }
        }
        self.length = next;
        Ok(())
    }
}

impl Drop for ExactStringWriter {
    fn drop(&mut self) {
        if self.capacity != 0 {
            let layout = Layout::array::<u8>(self.capacity)
                .expect("constructor checked exact String layout");
            // SAFETY: finish suppresses this Drop only after transferring the
            // same pointer/layout to String. Otherwise this remains sole owner.
            unsafe { dealloc(self.pointer.as_ptr(), layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::{
        HostEffectLiveAllocationKind, HostEffectStageReservation, ManagedHostEffectLedger,
    };

    #[test]
    fn exact_strings_share_one_claim_through_error_drop() {
        let ledger = ManagedHostEffectLedger::new(256);
        let stage = ledger.reserve_stage(256).expect("stage");
        let claim = stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, 6)
            .expect("single diagnostic claim");
        let mut owner = ExactHostEffectDiagnosticStrings::new(6, claim);
        let message = owner.render(3, "hé").expect("three-byte message");
        let identifier = owner.copy("id").expect("two-byte identifier");
        let builtin = owner.copy("r").expect("one-byte builtin");
        assert_eq!(message.capacity(), 3);
        assert_eq!(identifier.capacity(), 2);
        assert_eq!(builtin.capacity(), 1);
        let claim = owner.finish().expect("all payload bytes consumed");
        let error = runmat_async::runtime_error(message)
            .with_identifier(identifier)
            .with_builtin(builtin)
            .with_retained_diagnostic_allocation(claim.into_diagnostic_retention())
            .build();
        assert_eq!(error.message(), "hé");
        assert_eq!(error.identifier(), Some("id"));
        assert_eq!(error.context.builtin.as_deref(), Some("r"));
        let claimed = 6 + stage.live_claim_metadata_bytes();
        assert_eq!(ledger.snapshot().live_claimed_bytes, claimed as u64);
        drop(stage);
        assert_eq!(ledger.snapshot().live_claimed_bytes, claimed as u64);
        drop(error);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn allocation_or_length_failure_refunds_single_claim_without_partial_text() {
        let ledger = ManagedHostEffectLedger::new(256);
        let stage = ledger.reserve_stage(256).expect("stage");
        let claim = stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, 2)
            .expect("claim");
        let mut owner = ExactHostEffectDiagnosticStrings::new(2, claim);
        assert!(matches!(
            owner.copy("three"),
            Err(HostEffectAllocationError::ExceedsStageEnvelope)
        ));
        drop(owner);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
