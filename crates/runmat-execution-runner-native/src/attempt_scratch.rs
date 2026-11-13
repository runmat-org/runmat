//! Disjoint logical storage reservations for one original process attempt.
//! Filesystem blocks, inodes and arbitrary foreign-code writes are not OS quotas.
use runmat_execution::console::ConsoleLimits;
use runmat_execution_artifact::{ExecutionBundle, ObjectNamespace};

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) const CONSOLE_HEAD_BYTES: u64 = 4096;

pub(crate) fn console_bytes(limits: Option<ConsoleLimits>) -> NativeExecutionResult<u64> {
    let Some(limits) = limits else { return Ok(0) };
    limits.validate().map_err(invalid)?;
    // Worst-case JSON escaping and per-record metadata, plus old and new heads
    // during atomic publication. These are logical bytes, not filesystem RSS.
    limits
        .max_text_bytes
        .checked_mul(6)
        .and_then(|text| {
            u64::from(limits.max_records)
                .checked_mul(512)
                .and_then(|records| text.checked_add(records))
        })
        .and_then(|bytes| bytes.checked_add(CONSOLE_HEAD_BYTES * 2))
        .ok_or_else(|| invalid("console journal scratch size overflows"))
}

pub(crate) fn materialized_bytes(
    bundles: &[Option<ExecutionBundle>],
) -> NativeExecutionResult<u64> {
    bundles.iter().flatten().try_fold(0u64, |total, bundle| {
        bundle
            .objects
            .iter()
            .filter(|object| object.descriptor.namespace == ObjectNamespace::ForeignArtifact)
            .try_fold(total, |sum, object| {
                sum.checked_add(u64::try_from(object.bytes.len()).map_err(invalid)?)
                    .ok_or_else(|| invalid("Graph materialized scratch size overflows"))
            })
    })
}

/// Storage left for a new purpose after all existing original obligations.
/// Callers may partition this remainder, but cannot enlarge the admitted grant.
pub(crate) fn remaining_bytes(
    grant: u64,
    materialized: u64,
    console: Option<ConsoleLimits>,
) -> NativeExecutionResult<u64> {
    let required = materialized
        .checked_add(console_bytes(console)?)
        .ok_or_else(|| invalid("combined attempt scratch size overflows"))?;
    grant.checked_sub(required).ok_or_else(|| invalid(
        "original task scratch grant is below its combined materialization and console-journal reservation",
    ))
}

fn invalid(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn individually_fitting_reservations_cannot_spend_the_same_grant() {
        let limits = ConsoleLimits {
            max_records: 1,
            max_text_bytes: 4,
        };
        let console = console_bytes(Some(limits)).unwrap();
        assert!(remaining_bytes(console, 1, Some(limits)).is_err());
        assert_eq!(remaining_bytes(console + 17, 5, Some(limits)).unwrap(), 12);
        assert!(remaining_bytes(u64::MAX, u64::MAX, Some(limits)).is_err());
        assert_eq!(remaining_bytes(9, 4, None).unwrap(), 5);
    }
}
