//! Linear, metadata-only reconstruction of canonical one-sample checkpoint transitions.
use super::{
    extend_execution_prefix_digest, extend_observed_prefix_digest, initial_execution_prefix_digest,
    initial_observed_prefix_digest, ObservationError, ObservationHistoryCheckpointIdentity,
    ObservationHistoryCheckpointManifest, ObservationHistoryEntry,
    ObservationHistoryStreamIdentity, ObservationPartitionEvidence,
};
use runmat_execution::Digest;

type CheckpointPrefixIdentity = (
    Option<ObservationHistoryCheckpointIdentity>,
    u64,
    ObservationHistoryCheckpointIdentity,
);

impl ObservationHistoryCheckpointManifest {
    /// Reconstruct canonical one-sample transitions from a fully retained checkpoint.
    /// Returns (predecessor, zero-based sample ordinal, output checkpoint identity).
    /// Two linear scans validate the terminal link before exposing the iterator; no sample
    /// prefixes or numerical payloads are cloned. General checkpoint validation still permits
    /// batched checkpoints, but they cannot claim this one-task-per-sample lineage.
    pub(crate) fn canonical_prefix_identities(
        &self,
    ) -> Result<
        impl Iterator<Item = Result<CheckpointPrefixIdentity, ObservationError>> + '_,
        ObservationError,
    > {
        self.validate()?;
        if u64::try_from(self.samples.len()) != Ok(self.observed_sample_count) {
            return Err(ObservationError::Identity(
                "checkpoint prefix reconstruction requires fully retained samples",
            ));
        }
        let mut last = None;
        for transition in CheckpointPrefixIdentities::new(self)? {
            last = Some(transition?);
        }
        if last.is_none_or(|(predecessor, _, identity)| {
            predecessor != self.predecessor || identity != self.identity
        }) {
            return Err(ObservationError::Identity(
                "checkpoint terminal predecessor differs from canonical sample transitions",
            ));
        }
        CheckpointPrefixIdentities::new(self)
    }
}

struct CheckpointPrefixIdentities<'a> {
    stream: ObservationHistoryStreamIdentity,
    samples: std::iter::Enumerate<std::slice::Iter<'a, ObservationHistoryEntry>>,
    evidence: std::slice::Iter<'a, ObservationPartitionEvidence>,
    initial_execution_prefix: Digest,
    execution_prefix: Digest,
    initial_prefix: Digest,
    prefix: Digest,
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
}

impl<'a> CheckpointPrefixIdentities<'a> {
    fn new(manifest: &'a ObservationHistoryCheckpointManifest) -> Result<Self, ObservationError> {
        let prefix = initial_observed_prefix_digest(manifest.stream.identity)?;
        let execution_prefix = initial_execution_prefix_digest(manifest.stream.identity)?;
        Ok(Self {
            stream: manifest.stream.identity,
            samples: manifest.samples.iter().enumerate(),
            evidence: manifest.partition_evidence.iter(),
            initial_execution_prefix: execution_prefix,
            execution_prefix,
            initial_prefix: prefix,
            prefix,
            predecessor: None,
        })
    }
}

impl Iterator for CheckpointPrefixIdentities<'_> {
    type Item = Result<CheckpointPrefixIdentity, ObservationError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (ordinal, sample) = self.samples.next()?;
        Some((|| {
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| ObservationError::Limit("history sample ordinal"))?;
            let count = ordinal
                .checked_add(1)
                .ok_or(ObservationError::Limit("history sample count"))?;
            self.prefix = extend_observed_prefix_digest(self.prefix, sample.sample)?;
            let evidence = self.evidence.next().ok_or(ObservationError::Identity(
                "checkpoint execution evidence is absent",
            ))?;
            self.execution_prefix = extend_execution_prefix_digest(
                self.execution_prefix,
                sample.sample,
                evidence.identity,
            )?;
            let predecessor = self.predecessor;
            let identity = ObservationHistoryCheckpointIdentity::from_prefixes(
                self.stream,
                predecessor,
                count,
                count,
                self.initial_prefix,
                self.prefix,
                (self.initial_execution_prefix, self.execution_prefix),
            )?;
            self.predecessor = Some(identity);
            Ok((predecessor, ordinal, identity))
        })())
    }
}
