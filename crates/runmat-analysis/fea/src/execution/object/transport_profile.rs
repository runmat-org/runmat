//! Checked output-closure projection; does not allocate hypothetical chunks or descriptors.
use super::*;
use runmat_canonical_codec::{profile_serialized_shape, SerializedCanonicalProfile};
use serde::ser::SerializeSeq;

/// Conservative wire bounds, not a complete retained-heap or process-memory reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnalysisTransportProfile {
    pub partition_count: u64,
    pub chunk_count_upper_bound: u64,
    pub maximum_partition_chunks_upper_bound: u64,
    pub payload_bytes_upper_bound: u64,
    pub largest_chunk_bytes_upper_bound: u64,
    /// Requested borrowed-key slots used by manifest validation, not canonical encoding scratch.
    pub manifest_validation_scratch_bytes: u64,
    pub root: SerializedCanonicalProfile,
    pub total_bytes_upper_bound: u64,
}

impl AnalysisTransportProfile {
    /// Requested Rust heap for the builder: owned plan capacity, output inventories and bytes,
    /// one partition's staging-vector slots, and manifest validation slots. Excludes caller
    /// source buffers, root encoding, LogicalObject conversion, allocator overhead and RSS.
    /// The caller must reserve the stated plan capacity; no hypothetical identities are needed.
    pub fn builder_resident_bytes(&self, plan_capacity: usize) -> AnalysisExecutionResult<u64> {
        let partitions = usize::try_from(self.partition_count).map_err(|_| overflow())?;
        if plan_capacity < partitions {
            return Err(AnalysisExecutionError::Invalid(
                "plan capacity must cover the projected partition inventory".into(),
            ));
        }
        let chunks = usize::try_from(self.chunk_count_upper_bound).map_err(|_| overflow())?;
        let staging =
            usize::try_from(self.maximum_partition_chunks_upper_bound).map_err(|_| overflow())?;
        [
            plan::slots::<AnalysisPayloadPartitionPlan>(plan_capacity)?,
            plan::slots::<AnalysisChunkDescriptor>(chunks)?,
            plan::slots::<Vec<u8>>(chunks)?,
            plan::slots::<AnalysisPayloadPartitionDescriptor>(partitions)?,
            plan::slots::<Vec<u8>>(staging)?,
            self.payload_bytes_upper_bound,
            self.manifest_validation_scratch_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total.checked_add(bytes).ok_or_else(overflow)
        })
    }
}

/// Feed each independently encoded semantic partition, including the domain result header.
pub struct AnalysisTransportProfiler {
    chunk_bytes: u64,
    partitions: u64,
    chunks: u64,
    maximum_partition_chunks: u64,
    payload_bytes: u64,
    largest_chunk: u64,
}

impl AnalysisTransportProfiler {
    pub fn new(chunk_bytes: usize) -> AnalysisExecutionResult<Self> {
        if chunk_bytes == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "chunk byte limit must be non-zero".into(),
            ));
        }
        Ok(Self {
            chunk_bytes: u64::try_from(chunk_bytes).map_err(|_| overflow())?,
            partitions: 0,
            chunks: 0,
            maximum_partition_chunks: 0,
            payload_bytes: 0,
            largest_chunk: 0,
        })
    }

    /// Bounds must include the canonical envelope. Each partition starts a new chunk interval.
    /// Rejection leaves the prior projection unchanged.
    pub fn push(&mut self, encoded_bytes_upper_bound: u64) -> AnalysisExecutionResult<()> {
        if encoded_bytes_upper_bound == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "partition byte bound must be non-zero".into(),
            ));
        }
        let partition_chunks = encoded_bytes_upper_bound.div_ceil(self.chunk_bytes);
        let chunks = self
            .chunks
            .checked_add(partition_chunks)
            .ok_or_else(overflow)?;
        let partitions = self.partitions.checked_add(1).ok_or_else(overflow)?;
        let bytes = self
            .payload_bytes
            .checked_add(encoded_bytes_upper_bound)
            .ok_or_else(overflow)?;
        if chunks > MAX_CHUNKS as u64 || partitions > MAX_CHUNKS as u64 {
            return Err(AnalysisExecutionError::Limit(
                "projected numerical object inventory",
            ));
        }
        self.chunks = chunks;
        self.maximum_partition_chunks = self.maximum_partition_chunks.max(partition_chunks);
        self.partitions = partitions;
        self.payload_bytes = bytes;
        self.largest_chunk = self
            .largest_chunk
            .max(encoded_bytes_upper_bound.min(self.chunk_bytes));
        Ok(())
    }

    pub fn finish(
        self,
        kind: AnalysisNumericalObjectKind,
    ) -> AnalysisExecutionResult<AnalysisTransportProfile> {
        if self.partitions == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "transport profile requires partitions".into(),
            ));
        }
        let digest = Digest::from_bytes([0; 32]);
        // Integer values have fixed conservative widths in the shared profiler. The actual
        // descriptor types and shared root serializer own all map keys and wire structure.
        let root = profile_serialized_shape(
            &serialization::ManifestSerialization {
                schema_version: 1,
                kind,
                logical_identity: digest,
                payload_digest: digest,
                encoded_length: self.payload_bytes,
                chunks: Repeated {
                    value: AnalysisChunkDescriptor {
                        ordinal: 0,
                        digest,
                        encoded_length: 1,
                    },
                    count: self.chunks as usize,
                },
                partitions: Repeated {
                    value: AnalysisPayloadPartitionDescriptor {
                        ordinal: 0,
                        logical_identity: digest,
                        payload_digest: digest,
                        encoded_length: 1,
                        first_chunk_ordinal: 0,
                        chunk_count: 1,
                    },
                    count: self.partitions as usize,
                },
            },
            MANIFEST_LIMITS,
        )?;
        Ok(AnalysisTransportProfile {
            partition_count: self.partitions,
            chunk_count_upper_bound: self.chunks,
            maximum_partition_chunks_upper_bound: self.maximum_partition_chunks,
            payload_bytes_upper_bound: self.payload_bytes,
            largest_chunk_bytes_upper_bound: self.largest_chunk,
            manifest_validation_scratch_bytes: AnalysisObjectManifest::validation_scratch_bytes(
                self.partitions as usize,
            )?,
            total_bytes_upper_bound: self
                .payload_bytes
                .checked_add(root.encoded_bytes_upper_bound)
                .ok_or_else(overflow)?,
            root,
        })
    }
}

fn overflow() -> AnalysisExecutionError {
    AnalysisExecutionError::Limit("transport profile arithmetic overflow")
}

struct Repeated<T> {
    value: T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.count))?;
        for _ in 0..self.count {
            seq.serialize_element(&self.value)?;
        }
        seq.end()
    }
}

#[cfg(test)]
mod tests;
