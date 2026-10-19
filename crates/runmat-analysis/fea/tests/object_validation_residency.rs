//! Isolated requested-heap evidence for numerical-manifest uniqueness scratch.
#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisPayloadBuilder,
    AnalysisPayloadPartition, AnalysisPayloadPartitionPlan, AnalysisPayloadPlan,
};
use runmat_execution::Digest;

#[test]
fn manifest_validation_uses_only_reserved_borrowed_slots() {
    for count in [1_u64, 64, 4096] {
        let plan = AnalysisPayloadPlan::new(
            AnalysisNumericalObjectKind::SolverResult,
            1024,
            (0..count)
                .map(|ordinal| AnalysisPayloadPartitionPlan {
                    logical_identity: Digest::sha256(ordinal.to_le_bytes()),
                    encoded_bytes_upper_bound: b"partition".len() as u64,
                })
                .collect(),
        )
        .unwrap();
        let mut builder =
            AnalysisPayloadBuilder::new(Digest::sha256(b"validation-memory"), plan).unwrap();
        for ordinal in 0..count {
            builder
                .push(AnalysisPayloadPartition {
                    logical_identity: Digest::sha256(ordinal.to_le_bytes()),
                    payload: b"partition",
                })
                .unwrap();
        }
        let prepared = builder.finish().unwrap();
        let before = prepared.manifest.clone();
        let reserved = AnalysisObjectManifest::validation_scratch_bytes(count as usize).unwrap();
        let baseline = peak_allocator::begin_measurement();
        let validation = prepared.manifest.validate();
        let peak = peak_allocator::end_measurement().saturating_sub(baseline);
        validation.unwrap();
        assert!(
            peak <= reserved,
            "partitions={count} peak={peak} reserved={reserved}"
        );
        assert_eq!(prepared.manifest, before);
        println!("partitions={count} peak={peak} reserved={reserved}");
    }
    println!(
        "process_peak_rss_bytes={:?}",
        peak_allocator::process_peak_rss_bytes()
    );
}
