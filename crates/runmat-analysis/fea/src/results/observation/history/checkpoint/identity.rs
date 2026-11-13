//! Checkpoint identities commit validated incremental prefixes through constant-size headers.
use super::{
    ObservationError, ObservationHistoryCheckpointIdentity, ObservationHistoryCheckpointManifest,
    ObservationHistoryStreamIdentity, CHECKPOINT_PREFIX, CHECKPOINT_SCHEMA_VERSION,
    OBSERVATION_LIMITS,
};
use super::{ObservationHistorySampleIdentity, ObservationPartitionEvidenceIdentity};
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
        execution_prefixes: (Digest, Digest),
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
                execution_prefixes,
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
        (
            manifest.retained_execution_prefix_digest,
            manifest.observed_execution_prefix_digest,
        ),
    )
}

pub(in super::super) fn initial_execution_prefix_digest(
    stream: ObservationHistoryStreamIdentity,
) -> Result<Digest, ObservationError> {
    Ok(Digest::sha256(encode_contract(
        CHECKPOINT_PREFIX,
        "execution-prefix-initial",
        &stream,
        OBSERVATION_LIMITS,
    )?))
}

pub(in super::super) fn extend_execution_prefix_digest(
    prefix: Digest,
    sample: ObservationHistorySampleIdentity,
    evidence: ObservationPartitionEvidenceIdentity,
) -> Result<Digest, ObservationError> {
    Ok(Digest::sha256(encode_contract(
        CHECKPOINT_PREFIX,
        "execution-prefix-next",
        &(prefix, sample, evidence),
        OBSERVATION_LIMITS,
    )?))
}

pub(super) fn validate_execution_prefix(
    manifest: &ObservationHistoryCheckpointManifest,
) -> Result<(), ObservationError> {
    if manifest.partition_evidence.len() != manifest.samples.len()
        || (manifest.observed_sample_count == manifest.samples.len() as u64
            && manifest.retained_execution_prefix_digest
                != initial_execution_prefix_digest(manifest.stream.identity)?)
    {
        return Err(ObservationError::Identity(
            "checkpoint execution evidence coverage or initial prefix differs",
        ));
    }
    let mut prefix = manifest.retained_execution_prefix_digest;
    for (entry, evidence) in manifest.samples.iter().zip(&manifest.partition_evidence) {
        evidence.validate()?;
        if evidence.sample != entry.coordinate
            || evidence.event_attribution != entry.event_attribution
        {
            return Err(ObservationError::Identity(
                "checkpoint execution evidence coordinate differs",
            ));
        }
        prefix = extend_execution_prefix_digest(prefix, entry.sample, evidence.identity)?;
    }
    if prefix != manifest.observed_execution_prefix_digest {
        return Err(ObservationError::Identity(
            "checkpoint execution prefix does not converge",
        ));
    }
    Ok(())
}
