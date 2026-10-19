//! One full-text numeric-token diagnostic for floating and integer replay.
//! The staged caller uses its checked byte length for an exact retained
//! allocation; ordinary `to_string` below does not certify resource use.

use super::super::diagnostic::DiagnosticReservation;
use crate::context::host_effect::{ExactHostEffectDiagnosticStrings, HostEffectAllocationError};
use runmat_builtins::READMATRIX_ERROR_NON_NUMERIC_TOKEN;
use std::fmt;
use std::fmt::Write as _;

const BUILTIN: &str = "readmatrix";

const PREFIX: &str = "readmatrix: unable to parse numeric value '";
const LINE: &str = "' on line ";
const COLUMN: &str = " column ";

pub(super) struct NumericTokenDiagnostic<'a> {
    token: &'a str,
    line: usize,
    column: usize,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum NumericDiagnosticBuildError {
    #[error("numeric diagnostic length exceeds addressable storage")]
    LengthOverflow,
    #[error("numeric diagnostic allocation failed")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("numeric diagnostic formatting failed")]
    Formatting,
}

impl<'a> NumericTokenDiagnostic<'a> {
    pub(super) fn maximum_total_bytes(opened_extent: usize) -> Option<usize> {
        let coordinate = opened_extent.checked_add(1)?;
        PREFIX
            .len()
            .checked_add(opened_extent)?
            .checked_add(LINE.len())?
            .checked_add(decimal_digits(coordinate))?
            .checked_add(COLUMN.len())?
            .checked_add(decimal_digits(coordinate))?
            .checked_add(
                READMATRIX_ERROR_NON_NUMERIC_TOKEN
                    .identifier
                    .map_or(0, str::len),
            )?
            .checked_add(BUILTIN.len())
    }

    pub(super) fn new(token: &'a str, line: usize, column: usize) -> Self {
        Self {
            token,
            line,
            column,
        }
    }

    pub(super) fn requested_text_bytes(&self) -> Option<usize> {
        PREFIX
            .len()
            .checked_add(self.token.len())?
            .checked_add(LINE.len())?
            .checked_add(decimal_digits(self.line))?
            .checked_add(COLUMN.len())?
            .checked_add(decimal_digits(self.column))
    }

    pub(super) fn try_render(&self) -> Result<String, NumericDiagnosticBuildError> {
        let requested = self
            .requested_text_bytes()
            .ok_or(NumericDiagnosticBuildError::LengthOverflow)?;
        let mut text = String::new();
        text.try_reserve_exact(requested)
            .map_err(NumericDiagnosticBuildError::Allocation)?;
        write!(&mut text, "{self}").map_err(|_| NumericDiagnosticBuildError::Formatting)?;
        debug_assert_eq!(text.len(), requested);
        Ok(text)
    }

    pub(super) fn try_render_charged(
        &self,
        reservation: &dyn DiagnosticReservation,
    ) -> Result<crate::RuntimeError, HostEffectAllocationError> {
        let message_bytes = self
            .requested_text_bytes()
            .ok_or(HostEffectAllocationError::Overflow)?;
        let identifier = READMATRIX_ERROR_NON_NUMERIC_TOKEN.identifier;
        let total = message_bytes
            .checked_add(identifier.map_or(0, str::len))
            .and_then(|bytes| bytes.checked_add(BUILTIN.len()))
            .ok_or(HostEffectAllocationError::Overflow)?;
        let claim = reservation.reserve_diagnostic_claim(total)?;
        let strings = ExactHostEffectDiagnosticStrings::render(
            total,
            message_bytes,
            self,
            identifier,
            BUILTIN,
            claim,
        )?;
        Ok(strings.into_error())
    }
}

impl fmt::Display for NumericTokenDiagnostic<'_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "{PREFIX}{}{LINE}{}{COLUMN}{}",
            self.token, self.line, self.column
        )
    }
}

fn decimal_digits(mut number: usize) -> usize {
    let mut digits = 1;
    while number >= 10 {
        number /= 10;
        digits += 1;
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::{
        HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind, HostEffectStageReservation,
        ManagedHostEffectLedger, ManagedHostEffectStageReservation,
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
    fn exact_text_length_and_spelling_cover_unicode_and_wide_coordinates() {
        for (token, line, column) in [
            ("?", 1, 2),
            ("not-a-number", 100, 20_000),
            ("α☃", usize::MAX, usize::MAX),
        ] {
            let diagnostic = NumericTokenDiagnostic::new(token, line, column);
            let actual = diagnostic.try_render().expect("bounded diagnostic");
            assert_eq!(diagnostic.requested_text_bytes(), Some(actual.len()));
            assert_eq!(
                actual,
                format!(
                    "readmatrix: unable to parse numeric value '{}' on line {} column {}",
                    token, line, column
                )
            );
        }
    }

    #[test]
    fn float_and_integer_errors_retain_full_charged_text_code_builtin_until_last_drop() {
        let ledger = ManagedHostEffectLedger::new(4096);
        let stage = ledger.reserve_stage(4096).expect("diagnostic envelope");
        let reservation = TestReservation(stage.clone());
        let options = super::super::ReadMatrixOptions::default();
        for integer in [false, true] {
            let error = if integer {
                super::super::parse_integer_numeric_token_with_reservation(
                    "αbad",
                    &options,
                    runmat_value::NumericDType::U64,
                    12,
                    options.column_from_source(33),
                    Some(&reservation),
                )
                .expect_err("invalid integer token")
            } else {
                super::super::parse_numeric_token_with_reservation(
                    "αbad",
                    &options,
                    12,
                    options.column_from_source(33),
                    Some(&reservation),
                )
                .expect_err("invalid numeric token")
            };
            assert_eq!(
                error.message(),
                "readmatrix: unable to parse numeric value 'αbad' on line 12 column 34"
            );
            assert_eq!(
                error.identifier(),
                READMATRIX_ERROR_NON_NUMERIC_TOKEN.identifier
            );
            assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN));
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
    fn maximum_opened_extent_plan_covers_full_error_and_fixed_metadata() {
        let opened_extent = "αbad,1\n".len();
        let actual = NumericTokenDiagnostic::new("αbad", 1, 1)
            .requested_text_bytes()
            .unwrap()
            + READMATRIX_ERROR_NON_NUMERIC_TOKEN
                .identifier
                .map_or(0, str::len)
            + BUILTIN.len();
        assert!(NumericTokenDiagnostic::maximum_total_bytes(opened_extent).unwrap() >= actual);
    }
}
