//! Ordered semantic inventory and checked storage reservation for incremental construction.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnalysisPayloadPartitionPlan {
    pub logical_identity: Digest,
    pub encoded_bytes_upper_bound: u64,
}

/// Consumed by the builder; no tree or growing identity inventory is needed during encoding.
pub struct AnalysisPayloadPlan {
    pub(super) kind: AnalysisNumericalObjectKind,
    pub(super) chunk_bytes: usize,
    pub(super) partitions: Vec<AnalysisPayloadPartitionPlan>,
    transport: AnalysisTransportProfile,
}

impl AnalysisPayloadPlan {
    /// Allocate entry slots only after checking the shared inventory cardinality limit.
    pub fn reserve_partitions(
        count: usize,
    ) -> AnalysisExecutionResult<Vec<AnalysisPayloadPartitionPlan>> {
        check_partition_count(count)?;
        reserved(count)
    }

    pub fn new(
        kind: AnalysisNumericalObjectKind,
        chunk_bytes: usize,
        partitions: Vec<AnalysisPayloadPartitionPlan>,
    ) -> AnalysisExecutionResult<Self> {
        let mut profiler = AnalysisTransportProfiler::new(chunk_bytes)?;
        check_partition_count(partitions.len())?;
        for partition in &partitions {
            profiler.push(partition.encoded_bytes_upper_bound)?;
        }
        let transport = profiler.finish(kind)?;
        let mut identities = reserved::<&Digest>(partitions.len())?;
        identities.extend(partitions.iter().map(|entry| &entry.logical_identity));
        identities.sort_unstable();
        if identities.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(AnalysisExecutionError::Invalid(
                "planned partition identities must be unique".into(),
            ));
        }
        drop(identities);
        let plan = Self {
            kind,
            chunk_bytes,
            partitions,
            transport,
        };
        plan.resident_bytes()?;
        Ok(plan)
    }

    pub fn transport_profile(&self) -> &AnalysisTransportProfile {
        &self.transport
    }

    /// Requested Rust heap: owned plan capacity, reserved output inventories, accumulated bytes,
    /// one partition's staging-vector slots and identity-validation slots. Excludes caller source
    /// buffers, root encoding, LogicalObject conversion, allocator overhead and process RSS.
    pub fn resident_bytes(&self) -> AnalysisExecutionResult<u64> {
        self.transport
            .builder_resident_bytes(self.partitions.capacity())
    }
}

fn check_partition_count(count: usize) -> AnalysisExecutionResult<()> {
    if count == 0 || count > MAX_CHUNKS {
        Err(AnalysisExecutionError::Limit("planned partition inventory"))
    } else {
        Ok(())
    }
}

pub(super) fn slots<T>(count: usize) -> AnalysisExecutionResult<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(AnalysisExecutionError::Limit("planned inventory slots"))
}

pub(super) fn reserved<T>(count: usize) -> AnalysisExecutionResult<Vec<T>> {
    slots::<T>(count)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| AnalysisExecutionError::Limit("cannot reserve planned inventory"))?;
    Ok(values)
}

#[cfg(test)]
mod tests;
