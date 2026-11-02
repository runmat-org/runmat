use super::{OutputResult, RawOutputError};
use runmat_execution::console::ConsoleLimits;

pub(super) const METADATA_FILE_BYTES: u64 = 4096;
// One immutable identity and old/new heads during atomic replacement. The two
// stream files contain only raw bytes; they have no per-chunk storage overhead.
const METADATA_BYTES: u64 = METADATA_FILE_BYTES * 3;
const LOCAL_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawOutputBudget {
    payload_bytes: u64,
}

impl RawOutputBudget {
    /// Partition the original grant after all other logical scratch obligations.
    /// This does not grant additional storage or enforce an OS filesystem quota.
    pub(crate) fn from_grant(
        grant: u64,
        materialized: u64,
        console: Option<ConsoleLimits>,
    ) -> OutputResult<Self> {
        let remaining = crate::attempt_scratch::remaining_bytes(grant, materialized, console)?;
        let payload_bytes =
            remaining
                .checked_sub(METADATA_BYTES)
                .ok_or(RawOutputError::StorageLimit {
                    required: METADATA_BYTES,
                    maximum: remaining,
                })?;
        Ok(Self { payload_bytes })
    }

    /// Quote synthesized local tasks before admission. An original caller grant
    /// must use `from_grant`; it cannot be enlarged with this constructor.
    pub(crate) fn local_scratch_bytes(
        materialized: u64,
        console: Option<ConsoleLimits>,
    ) -> OutputResult<u64> {
        materialized
            .checked_add(crate::attempt_scratch::console_bytes(console)?)
            .and_then(|bytes| bytes.checked_add(METADATA_BYTES))
            .and_then(|bytes| bytes.checked_add(LOCAL_OUTPUT_BYTES))
            .ok_or(RawOutputError::Invalid("local scratch quote overflows"))
    }

    pub(crate) fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }
}
