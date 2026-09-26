//! Incremental shared numerical-object chunking; partition inputs are never retained.
use std::collections::BTreeSet;

use super::*;

/// Builds the shared canonical chunk inventory one semantic payload at a time. Final chunks
/// remain resident until ownership is transferred by `finish`; this is not a constant-memory sink.
pub struct AnalysisPayloadBuilder {
    kind: AnalysisNumericalObjectKind,
    logical_identity: Digest,
    chunk_bytes: usize,
    payload_hash: Sha256,
    encoded_length: u64,
    chunks: Vec<AnalysisChunkDescriptor>,
    payload_chunks: Vec<Vec<u8>>,
    partitions: Vec<AnalysisPayloadPartitionDescriptor>,
    identities: BTreeSet<Digest>,
}

impl AnalysisPayloadBuilder {
    pub fn new(
        kind: AnalysisNumericalObjectKind,
        logical_identity: Digest,
        chunk_bytes: usize,
    ) -> AnalysisExecutionResult<Self> {
        if chunk_bytes == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "chunk byte limit must be non-zero".into(),
            ));
        }
        Ok(Self {
            kind,
            logical_identity,
            chunk_bytes,
            payload_hash: Sha256::new(),
            encoded_length: 0,
            chunks: Vec::new(),
            payload_chunks: Vec::new(),
            partitions: Vec::new(),
            identities: BTreeSet::new(),
        })
    }

    /// Copies one partition into final transport chunks. Rejected inputs leave prior content
    /// unchanged; callers may discard the input immediately after a successful push.
    pub fn push(&mut self, partition: AnalysisPayloadPartition<'_>) -> AnalysisExecutionResult<()> {
        if partition.payload.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "numerical object partitions must be non-empty".into(),
            ));
        }
        if self.identities.contains(&partition.logical_identity) {
            return Err(AnalysisExecutionError::Invalid(
                "numerical object partition identities must be unique".into(),
            ));
        }
        let (chunk_count, encoded_length) = checked_append(
            self.chunks.len(),
            self.partitions.len(),
            self.encoded_length,
            partition.payload.len(),
            self.chunk_bytes,
        )?;
        // All cardinality, conversion and byte-total checks precede hash/inventory mutation.
        let first_chunk_ordinal = self.chunks.len() as u32;
        self.identities.insert(partition.logical_identity);
        for bytes in partition.payload.chunks(self.chunk_bytes) {
            self.chunks.push(AnalysisChunkDescriptor {
                ordinal: self.chunks.len() as u32,
                digest: Digest::sha256(bytes),
                encoded_length: bytes.len() as u64,
            });
            self.payload_chunks.push(bytes.to_vec());
        }
        self.partitions.push(AnalysisPayloadPartitionDescriptor {
            ordinal: self.partitions.len() as u32,
            logical_identity: partition.logical_identity,
            payload_digest: Digest::sha256(partition.payload),
            encoded_length: partition.payload.len() as u64,
            first_chunk_ordinal,
            chunk_count: chunk_count as u32,
        });
        self.payload_hash.update(partition.payload);
        self.encoded_length = encoded_length;
        Ok(())
    }

    pub fn finish(self) -> AnalysisExecutionResult<ChunkedAnalysisPayload> {
        if self.partitions.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "numerical object partitions must be non-empty".into(),
            ));
        }
        Ok(ChunkedAnalysisPayload {
            manifest: AnalysisObjectManifest {
                schema_version: 1,
                kind: self.kind,
                logical_identity: self.logical_identity,
                payload_digest: Digest::from_bytes(self.payload_hash.finalize().into()),
                encoded_length: self.encoded_length,
                chunks: self.chunks,
                partitions: self.partitions,
            },
            chunks: self.payload_chunks,
        })
    }
}

fn checked_append(
    chunks: usize,
    partitions: usize,
    encoded_length: u64,
    payload_length: usize,
    chunk_bytes: usize,
) -> AnalysisExecutionResult<(usize, u64)> {
    let count = payload_length.div_ceil(chunk_bytes);
    let total_chunks = chunks
        .checked_add(count)
        .ok_or(AnalysisExecutionError::Limit(
            "numerical object chunk count",
        ))?;
    let total_partitions = partitions
        .checked_add(1)
        .ok_or(AnalysisExecutionError::Limit(
            "numerical object partition count",
        ))?;
    if total_chunks > MAX_CHUNKS || total_chunks > u32::MAX as usize {
        return Err(AnalysisExecutionError::Limit(
            "numerical object chunk count",
        ));
    }
    if total_partitions > MAX_CHUNKS || total_partitions > u32::MAX as usize {
        return Err(AnalysisExecutionError::Limit(
            "numerical object partition count",
        ));
    }
    let bytes = u64::try_from(payload_length)
        .map_err(|_| AnalysisExecutionError::Limit("object byte length overflow"))?;
    let total_bytes = encoded_length
        .checked_add(bytes)
        .ok_or(AnalysisExecutionError::Limit("object byte length overflow"))?;
    Ok((count, total_bytes))
}

#[cfg(test)]
mod tests;
