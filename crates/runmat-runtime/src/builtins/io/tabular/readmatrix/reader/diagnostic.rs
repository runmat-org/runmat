//! Fixed late reader failures that can be charged before the opened-source scan.

use super::{admission, readmatrix_error_with, readmatrix_error_with_source, BUILTIN_NAME};
use crate::context::host_effect::{
    ExactHostEffectDiagnosticStrings, HostEffectAllocationError, HostEffectLiveAllocationClaim,
    HostEffectReceipt, HostEffectStagePermit,
};
use crate::RuntimeError;
use runmat_builtins::{
    BuiltinErrorDescriptor, READMATRIX_ERROR_IO_READ, READMATRIX_ERROR_TENSOR_BUILD,
};

pub(in crate::builtins::io::tabular::readmatrix::reader) trait DiagnosticReservation {
    fn reserve_diagnostic_claim(
        &self,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError>;
}

impl DiagnosticReservation for HostEffectStagePermit {
    fn reserve_diagnostic_claim(
        &self,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        HostEffectStagePermit::reserve_diagnostic_claim(self, bytes)
    }
}

impl DiagnosticReservation for HostEffectReceipt {
    fn reserve_diagnostic_claim(
        &self,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        HostEffectReceipt::reserve_diagnostic_claim(self, bytes)
    }
}

#[derive(Clone, Copy)]
pub(in crate::builtins::io::tabular::readmatrix::reader) enum FixedReaderDiagnostic {
    ChangedSource,
    InvalidUtf8,
    MatrixOverflow,
    LineOverflow,
    ChunkOverflow,
}

impl FixedReaderDiagnostic {
    fn parts(self) -> (&'static BuiltinErrorDescriptor, &'static str) {
        match self {
            Self::ChangedSource => (
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: opened source changed between validation and output",
            ),
            Self::InvalidUtf8 => (
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'",
            ),
            Self::MatrixOverflow => (
                &READMATRIX_ERROR_TENSOR_BUILD,
                "readmatrix: matrix dimensions exceed addressable storage",
            ),
            Self::LineOverflow => (
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: input line number exceeds platform limits",
            ),
            Self::ChunkOverflow => (
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: decoded input chunk exceeds the bounded reader capacity",
            ),
        }
    }

    pub(in crate::builtins::io::tabular::readmatrix::reader) fn render(
        self,
        reservation: Option<&dyn DiagnosticReservation>,
    ) -> RuntimeError {
        let (descriptor, message) = self.parts();
        let Some(reservation) = reservation else {
            return readmatrix_error_with(descriptor, message);
        };
        let result = (|| {
            let identifier = descriptor.identifier;
            let total = message
                .len()
                .checked_add(identifier.map_or(0, str::len))
                .and_then(|bytes| bytes.checked_add(BUILTIN_NAME.len()))
                .ok_or(HostEffectAllocationError::Overflow)?;
            let claim = reservation.reserve_diagnostic_claim(total)?;
            ExactHostEffectDiagnosticStrings::render(
                total,
                message.len(),
                message,
                identifier,
                BUILTIN_NAME,
                claim,
            )
            .map(ExactHostEffectDiagnosticStrings::into_error)
        })();
        result.unwrap_or_else(|source| admission::allocation_error(BUILTIN_NAME, source))
    }

    pub(in crate::builtins::io::tabular::readmatrix::reader) fn invalid_utf8_with_source(
        source: std::str::Utf8Error,
        reservation: Option<&dyn DiagnosticReservation>,
    ) -> RuntimeError {
        let (descriptor, message) = Self::InvalidUtf8.parts();
        let Some(reservation) = reservation else {
            return readmatrix_error_with_source(descriptor, message, source);
        };
        let result = (|| {
            let identifier = descriptor.identifier;
            let total = message
                .len()
                .checked_add(identifier.map_or(0, str::len))
                .and_then(|bytes| bytes.checked_add(BUILTIN_NAME.len()))
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<std::str::Utf8Error>()))
                .ok_or(HostEffectAllocationError::Overflow)?;
            let claim = reservation.reserve_diagnostic_claim(total)?;
            ExactHostEffectDiagnosticStrings::render_with_utf8_source(
                total,
                message,
                identifier,
                BUILTIN_NAME,
                source,
                claim,
            )
        })();
        result.unwrap_or_else(|source| admission::allocation_error(BUILTIN_NAME, source))
    }
}

pub(in crate::builtins::io::tabular::readmatrix::reader) fn maximum_fixed_diagnostic_bytes(
) -> Option<usize> {
    let mut maximum = 0;
    for kind in [
        FixedReaderDiagnostic::ChangedSource,
        FixedReaderDiagnostic::InvalidUtf8,
        FixedReaderDiagnostic::MatrixOverflow,
        FixedReaderDiagnostic::LineOverflow,
        FixedReaderDiagnostic::ChunkOverflow,
    ] {
        let (descriptor, message) = kind.parts();
        let source_bytes = if matches!(kind, FixedReaderDiagnostic::InvalidUtf8) {
            std::mem::size_of::<std::str::Utf8Error>()
        } else {
            0
        };
        let bytes = message
            .len()
            .checked_add(descriptor.identifier.map_or(0, str::len))?
            .checked_add(BUILTIN_NAME.len())?
            .checked_add(source_bytes)?;
        maximum = maximum.max(bytes);
    }
    Some(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::{
        HostEffectLiveAllocationKind, HostEffectStageReservation, ManagedHostEffectLedger,
        ManagedHostEffectStageReservation,
    };
    use std::sync::Arc;

    struct TestReservation(Arc<ManagedHostEffectStageReservation>);

    impl DiagnosticReservation for TestReservation {
        fn reserve_diagnostic_claim(
            &self,
            bytes: usize,
        ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
            self.0
                .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, bytes)
        }
    }

    #[test]
    fn fixed_errors_preserve_spelling_code_and_charge_through_last_owner() {
        let ledger = ManagedHostEffectLedger::new(4096);
        let stage = ledger.reserve_stage(4096).expect("stage");
        let reservation = TestReservation(stage.clone());
        for kind in [
            FixedReaderDiagnostic::ChangedSource,
            FixedReaderDiagnostic::InvalidUtf8,
            FixedReaderDiagnostic::MatrixOverflow,
            FixedReaderDiagnostic::LineOverflow,
            FixedReaderDiagnostic::ChunkOverflow,
        ] {
            let (descriptor, message) = kind.parts();
            let error = kind.render(Some(&reservation));
            assert_eq!(error.message(), message);
            assert_eq!(error.identifier(), descriptor.identifier);
            assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
            assert!(error.source.is_none());
            assert!(error.retained_diagnostic_allocation.is_some());
            let payload = error.message.capacity()
                + error.identifier.as_ref().map_or(0, String::capacity)
                + error.context.builtin.as_ref().unwrap().capacity();
            assert_eq!(
                ledger.snapshot().live_claimed_bytes,
                (payload + stage.live_claim_metadata_bytes()) as u64
            );
            drop(error);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        }
        drop(reservation);
        drop(stage);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn maximum_plan_covers_every_fixed_diagnostic() {
        let maximum = maximum_fixed_diagnostic_bytes().expect("fixed lengths");
        for kind in [
            FixedReaderDiagnostic::ChangedSource,
            FixedReaderDiagnostic::InvalidUtf8,
            FixedReaderDiagnostic::MatrixOverflow,
            FixedReaderDiagnostic::LineOverflow,
            FixedReaderDiagnostic::ChunkOverflow,
        ] {
            let (descriptor, message) = kind.parts();
            assert!(
                maximum
                    >= message.len()
                        + descriptor.identifier.map_or(0, str::len)
                        + BUILTIN_NAME.len()
            );
        }
    }

    #[test]
    fn cancelled_utf8_source_reservation_returns_typed_allocation_failure() {
        struct Cancelled;
        impl DiagnosticReservation for Cancelled {
            fn reserve_diagnostic_claim(
                &self,
                _bytes: usize,
            ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
                Err(HostEffectAllocationError::Cancelled)
            }
        }

        let bytes = vec![0xff];
        let utf8 = std::str::from_utf8(&bytes).unwrap_err();
        let error = FixedReaderDiagnostic::invalid_utf8_with_source(utf8, Some(&Cancelled));
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(error
            .source
            .as_deref()
            .is_some_and(|source| source.is::<HostEffectAllocationError>()));
        assert!(error.retained_diagnostic_allocation.is_none());
    }
}
