//! Isolated peak Rust heap for generated payloads and retained output chunks.
#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisPayloadBuilder, AnalysisPayloadPartition,
    AnalysisPayloadPartitionPlan, AnalysisPayloadPlan,
};
use runmat_execution::Digest;

#[test]
fn incremental_preparation_retains_output_plus_one_source_partition() {
    const PARTITION_BYTES: usize = 512 * 1024;
    const CHUNK_BYTES: usize = 64 * 1024;
    for count in [16_u64, 64] {
        let plan = AnalysisPayloadPlan::new(
            AnalysisNumericalObjectKind::SolverResult,
            CHUNK_BYTES,
            (0..count)
                .map(|ordinal| AnalysisPayloadPartitionPlan {
                    logical_identity: Digest::sha256(ordinal.to_le_bytes()),
                    encoded_bytes_upper_bound: PARTITION_BYTES as u64,
                })
                .collect(),
        )
        .unwrap();
        let reserved = plan.resident_bytes().unwrap();
        // Plan construction is outside measurement; its included slots make the bound conservative.
        let baseline = peak_allocator::begin_measurement();
        let mut builder =
            AnalysisPayloadBuilder::new(Digest::sha256(b"incremental-memory"), plan).unwrap();
        for ordinal in 0..count {
            let payload = vec![ordinal as u8; PARTITION_BYTES];
            builder
                .push(AnalysisPayloadPartition {
                    logical_identity: Digest::sha256(ordinal.to_le_bytes()),
                    payload: &payload,
                })
                .unwrap();
        }
        let prepared = builder.finish().unwrap();
        let growth = peak_allocator::end_measurement().saturating_sub(baseline);
        let output_bytes = count * PARTITION_BYTES as u64;
        // Governed builder heap plus one generated source; excludes field serialization.
        assert!(
            growth <= reserved + PARTITION_BYTES as u64,
            "output={output_bytes}, peak={growth}, builder_reservation={reserved}"
        );
        assert_eq!(prepared.manifest.encoded_length, output_bytes);
        prepared.revalidate(CHUNK_BYTES).unwrap();
        println!("partitions={count} output_bytes={output_bytes} peak_growth={growth} process_peak_rss_bytes={:?}", peak_allocator::process_peak_rss_bytes());
    }
}
