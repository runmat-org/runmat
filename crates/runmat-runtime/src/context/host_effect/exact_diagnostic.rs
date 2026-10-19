//! Exact physical String blocks under one already-admitted diagnostic claim.
//! This owner charges bytes, not a whole error or complete invocation.

use super::{HostEffectAllocationError, HostEffectLiveAllocationClaim};
use std::alloc::{alloc, dealloc, Layout};
use std::fmt::{self, Write as _};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

pub(crate) struct ExactHostEffectDiagnosticStrings {
    message: String,
    identifier: Option<String>,
    builtin: String,
    // Last field: every present physical String block drops before its charge.
    claim: HostEffectLiveAllocationClaim,
}

impl ExactHostEffectDiagnosticStrings {
    /// The claim must reserve exactly the checked sum of the present requested
    /// capacities. Construction retains it through every partial success.
    pub(crate) fn render(
        authorized_bytes: usize,
        message_bytes: usize,
        message: impl fmt::Display,
        identifier: Option<&str>,
        builtin: &str,
        claim: HostEffectLiveAllocationClaim,
    ) -> Result<Self, HostEffectAllocationError> {
        Self::render_with_extra_bytes(
            authorized_bytes,
            message_bytes,
            message,
            identifier,
            builtin,
            0,
            claim,
        )
    }

    fn render_with_extra_bytes(
        authorized_bytes: usize,
        message_bytes: usize,
        message: impl fmt::Display,
        identifier: Option<&str>,
        builtin: &str,
        extra_bytes: usize,
        claim: HostEffectLiveAllocationClaim,
    ) -> Result<Self, HostEffectAllocationError> {
        let total = message_bytes
            .checked_add(identifier.map_or(0, str::len))
            .and_then(|bytes| bytes.checked_add(builtin.len()))
            .and_then(|bytes| bytes.checked_add(extra_bytes))
            .ok_or(HostEffectAllocationError::Overflow)?;
        if total != authorized_bytes {
            return Err(HostEffectAllocationError::ExceedsStageEnvelope);
        }
        let mut output = ExactStringWriter::new(message_bytes)?;
        write!(&mut output, "{message}")
            .map_err(|_| HostEffectAllocationError::InvalidInvocation)?;
        let message = output.finish()?;
        let identifier = identifier
            .map(|identifier| {
                let mut output = ExactStringWriter::new(identifier.len())?;
                output
                    .write_str(identifier)
                    .map_err(|_| HostEffectAllocationError::InvalidInvocation)?;
                output.finish()
            })
            .transpose()?;
        let mut output = ExactStringWriter::new(builtin.len())?;
        output
            .write_str(builtin)
            .map_err(|_| HostEffectAllocationError::InvalidInvocation)?;
        let builtin = output.finish()?;
        Ok(Self {
            message,
            identifier,
            builtin,
            claim,
        })
    }

    pub(crate) fn into_error(self) -> crate::RuntimeError {
        let mut builder = crate::build_runtime_error(self.message)
            .with_builtin(self.builtin)
            .with_retained_diagnostic_allocation(self.claim.into_diagnostic_retention());
        if let Some(identifier) = self.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    /// The concrete `Utf8Error` has no dynamic payload. Reserve its actual
    /// compiled Box layout along with the strings and preserve its typed cause.
    pub(crate) fn render_with_utf8_source(
        authorized_bytes: usize,
        message: &'static str,
        identifier: Option<&str>,
        builtin: &str,
        source: std::str::Utf8Error,
        claim: HostEffectLiveAllocationClaim,
    ) -> Result<crate::RuntimeError, HostEffectAllocationError> {
        let layout = Layout::new::<std::str::Utf8Error>();
        if layout.size() == 0 {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let strings = Self::render_with_extra_bytes(
            authorized_bytes,
            message.len(),
            message,
            identifier,
            builtin,
            layout.size(),
            claim,
        )?;
        let pointer = NonNull::new(unsafe { alloc(layout) })
            .ok_or(HostEffectAllocationError::AllocationFailed)?;
        // SAFETY: the checked, unique global allocation has exactly the
        // layout of Utf8Error. Box takes over its matching deallocation.
        let boxed = unsafe {
            pointer.as_ptr().cast::<std::str::Utf8Error>().write(source);
            Box::from_raw(pointer.as_ptr().cast::<std::str::Utf8Error>())
        };
        let mut error = strings.into_error();
        error.source = Some(boxed);
        Ok(error)
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
        let owner = ExactHostEffectDiagnosticStrings::render(6, 3, "hé", Some("id"), "r", claim)
            .expect("all three exact blocks");
        assert_eq!(owner.message.capacity(), 3);
        assert_eq!(owner.identifier.as_ref().unwrap().capacity(), 2);
        assert_eq!(owner.builtin.capacity(), 1);
        let error = owner.into_error();
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
    fn absent_identifier_stays_absent_without_an_allocation_or_placeholder() {
        let ledger = ManagedHostEffectLedger::new(256);
        let stage = ledger.reserve_stage(256).expect("stage");
        let claim = stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, 4)
            .expect("message and builtin claim");
        let owner = ExactHostEffectDiagnosticStrings::render(4, 3, "hé", None, "r", claim)
            .expect("optional identifier");
        assert!(owner.identifier.is_none());
        let error = owner.into_error();
        assert_eq!(error.message(), "hé");
        assert_eq!(error.identifier(), None);
        assert_eq!(error.context.builtin.as_deref(), Some("r"));
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            (4 + stage.live_claim_metadata_bytes()) as u64
        );
        drop(stage);
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        drop(error);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn allocation_or_length_failure_refunds_single_claim_without_partial_text() {
        let ledger = ManagedHostEffectLedger::new(256);
        let stage = ledger.reserve_stage(256).expect("stage");
        let claim = stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, 2)
            .expect("claim");
        assert!(matches!(
            ExactHostEffectDiagnosticStrings::render(2, 3, "abc", None, "", claim),
            Err(HostEffectAllocationError::ExceedsStageEnvelope)
        ));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn formatting_mismatch_after_partial_write_refunds_without_escaping_text() {
        struct TooShort;
        impl fmt::Display for TooShort {
            fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
                output.write_str("a")
            }
        }

        let ledger = ManagedHostEffectLedger::new(256);
        let stage = ledger.reserve_stage(256).expect("stage");
        let claim = stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, 4)
            .expect("claim");
        assert!(matches!(
            ExactHostEffectDiagnosticStrings::render(4, 2, TooShort, Some("i"), "b", claim),
            Err(HostEffectAllocationError::InvalidInvocation)
        ));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn utf8_source_layout_mismatch_refunds_before_box_or_text_escapes() {
        let ledger = ManagedHostEffectLedger::new(256);
        let stage = ledger.reserve_stage(256).expect("stage");
        let claim = stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, 4)
            .expect("claim");
        let bytes = vec![0xff];
        let source = std::str::from_utf8(&bytes).unwrap_err();
        assert!(matches!(
            ExactHostEffectDiagnosticStrings::render_with_utf8_source(
                4,
                "a",
                Some("i"),
                "b",
                source,
                claim,
            ),
            Err(HostEffectAllocationError::ExceedsStageEnvelope)
        ));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
