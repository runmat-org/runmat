//! Checkpoint identities commit validated incremental prefixes through constant-size headers.
use super::{
    ObservationError, ObservationHistoryCheckpointIdentity, ObservationHistoryCheckpointManifest,
    ObservationHistoryStreamIdentity, CHECKPOINT_PREFIX, CHECKPOINT_SCHEMA_VERSION,
    OBSERVATION_LIMITS,
};
use runmat_canonical_codec::encode_contract;
use runmat_execution::Digest;

impl ObservationHistoryCheckpointIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }

    /// Constant-size checkpoint identity from validated stream and incremental sample
    /// commitments. Callers must validate entries and extend their prefix commitments;
    /// hashing these headers alone does not establish source or evicted-prefix provenance.
    pub(crate) fn from_prefixes(
        stream: ObservationHistoryStreamIdentity,
        predecessor: Option<Self>,
        observed_sample_count: u64,
        retained_sample_count: u64,
        retained_prefix_digest: Digest,
        observed_prefix_digest: Digest,
    ) -> Result<Self, ObservationError> {
        if observed_sample_count == 0
            || retained_sample_count == 0
            || retained_sample_count > observed_sample_count
        {
            return Err(ObservationError::Invalid(
                "history checkpoint identity has invalid sample counts".into(),
            ));
        }
        Ok(Self(Digest::sha256(encode_contract(
            CHECKPOINT_PREFIX,
            "identity",
            &(
                CHECKPOINT_SCHEMA_VERSION,
                stream,
                predecessor,
                observed_sample_count,
                retained_sample_count,
                retained_prefix_digest,
                observed_prefix_digest,
            ),
            OBSERVATION_LIMITS,
        )?)))
    }
}

pub(super) fn derive_checkpoint_identity(
    manifest: &ObservationHistoryCheckpointManifest,
) -> Result<ObservationHistoryCheckpointIdentity, ObservationError> {
    // Entry validation binds every coordinate, event and result to its sample identity;
    // prefix validation commits their exact order. Do not hash/copy the full prefix again.
    ObservationHistoryCheckpointIdentity::from_prefixes(
        manifest.stream.identity,
        manifest.predecessor,
        manifest.observed_sample_count,
        u64::try_from(manifest.samples.len())
            .map_err(|_| ObservationError::Limit("history checkpoint retained samples"))?,
        manifest.retained_prefix_digest,
        manifest.observed_prefix_digest,
    )
}
