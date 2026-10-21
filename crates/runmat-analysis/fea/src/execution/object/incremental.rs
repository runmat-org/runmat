//! Incremental shared numerical-object chunking; partition inputs are never retained.
use super::*;

/// Builds the shared canonical chunk inventory one semantic payload at a time. Final chunks
/// remain resident until ownership is transferred by `finish`; this is not a constant-memory sink.
pub struct AnalysisPayloadBuilder {
    plan: AnalysisPayloadPlan,
    logical_identity: Digest,
    chunk_bytes: usize,
    payload_hash: Sha256,
    encoded_length: u64,
    chunks: Vec<AnalysisChunkDescriptor>,
    payload_chunks: Vec<Vec<u8>>,
    partitions: Vec<AnalysisPayloadPartitionDescriptor>,
}

impl AnalysisPayloadBuilder {
    pub fn new(
        logical_identity: Digest,
        plan: AnalysisPayloadPlan,
    ) -> AnalysisExecutionResult<Self> {
        let chunks = usize::try_from(plan.transport_profile().chunk_count_upper_bound)
            .map_err(|_| AnalysisExecutionError::Limit("planned chunk count"))?;
        let partitions = plan.partitions.len();
        Ok(Self {
            logical_identity,
            chunk_bytes: plan.chunk_bytes,
            payload_hash: Sha256::new(),
            encoded_length: 0,
            chunks: plan::reserved(chunks)?,
            payload_chunks: plan::reserved(chunks)?,
            partitions: plan::reserved(partitions)?,
            plan,
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
        let expected = self.plan.partitions.get(self.partitions.len()).ok_or(
            AnalysisExecutionError::Limit("partition exceeds planned inventory"),
        )?;
        if expected.logical_identity != partition.logical_identity {
            return Err(AnalysisExecutionError::Invalid(
                "partition differs from next planned identity".into(),
            ));
        }
        if partition.payload.len() as u64 > expected.encoded_bytes_upper_bound {
            return Err(AnalysisExecutionError::Limit(
                "partition exceeds planned byte bound",
            ));
        }
        let (chunk_count, encoded_length) = checked_append(
            self.chunks.len(),
            self.partitions.len(),
            self.encoded_length,
            partition.payload.len(),
            self.chunk_bytes,
        )?;
        // Stage this partition fallibly before any mutation. Payload bytes move into already
        // reserved output inventories; staging requires only additional outer Vec slots.
        let mut staged = plan::reserved(chunk_count)?;
        for bytes in partition.payload.chunks(self.chunk_bytes) {
            let mut owned = plan::reserved(bytes.len())?;
            owned.extend_from_slice(bytes);
            staged.push(owned);
        }
        let first_chunk_ordinal = self.chunks.len() as u32;
        // Chunk digests are independent; hash them on the admitted pool when one is installed.
        let digests = runmat_cpu_execution::ordered_map(&staged, |bytes| Digest::sha256(bytes));
        for (bytes, digest) in staged.into_iter().zip(digests) {
            self.chunks.push(AnalysisChunkDescriptor {
                ordinal: self.chunks.len() as u32,
                digest,
                encoded_length: bytes.len() as u64,
            });
            self.payload_chunks.push(bytes);
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
        if self.partitions.len() != self.plan.partitions.len() {
            return Err(AnalysisExecutionError::Invalid(
                "numerical object omits planned partitions".into(),
            ));
        }
        Ok(ChunkedAnalysisPayload {
            manifest: AnalysisObjectManifest {
                schema_version: 1,
                kind: self.plan.kind,
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
