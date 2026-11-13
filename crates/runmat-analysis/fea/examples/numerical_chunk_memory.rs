//! Checks chunk preparation/verification heap growth, excluding borrowed input construction.
//! This measures Rust allocations, not native storage or process RSS attribution.
use std::error::Error;

#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use runmat_analysis_fea::execution::{
    chunk_partitioned_numerical_payload, AnalysisNumericalObjectKind, AnalysisPayloadPartition,
};
use runmat_execution::Digest;

fn main() -> Result<(), Box<dyn Error>> {
    const CHUNK_BYTES: usize = 64 * 1024;
    for partition_bytes in [512 * 1024, 2 * 1024 * 1024] {
        let inputs = (0_u8..16)
            .map(|value| vec![value; partition_bytes])
            .collect::<Vec<_>>();
        let partitions = inputs
            .iter()
            .enumerate()
            .map(|(ordinal, bytes)| AnalysisPayloadPartition {
                logical_identity: Digest::sha256(ordinal.to_le_bytes()),
                payload: bytes,
            })
            .collect::<Vec<_>>();
        let payload_bytes = (inputs.len() * partition_bytes) as u64;
        let baseline = peak_allocator::begin_measurement();
        let prepared = chunk_partitioned_numerical_payload(
            AnalysisNumericalObjectKind::SolverResult,
            Digest::sha256(b"memory-diagnostic"),
            &partitions,
            CHUNK_BYTES,
        );
        let preparation_growth = peak_allocator::end_measurement().saturating_sub(baseline);
        let prepared = prepared?;
        // Chunk bytes must remain owned, but no second payload-sized hashing buffer is needed.
        if preparation_growth > payload_bytes + 256 * 1024 {
            return Err(
                "preparation retained more than output bytes plus metadata allowance".into(),
            );
        }
        let baseline = peak_allocator::begin_measurement();
        let validation = prepared.revalidate(CHUNK_BYTES);
        let validation_growth = peak_allocator::end_measurement().saturating_sub(baseline);
        validation?;
        if validation_growth > 64 * 1024 {
            return Err("closure verification allocated more than metadata allowance".into());
        }
        println!("payload_bytes={payload_bytes} preparation_rust_heap_growth={preparation_growth} validation_rust_heap_growth={validation_growth} process_peak_rss_bytes={:?}", peak_allocator::process_peak_rss_bytes());
    }
    Ok(())
}
