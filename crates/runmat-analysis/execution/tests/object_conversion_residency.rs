//! Isolated requested Rust heap for planned chunk construction and artifact conversion.
#[path = "../../fea/support/peak_allocator/counting.rs"]
mod counting;

use runmat_analysis_execution::{
    prepare_partitioned_analysis_objects, projected_chunked_output_resource_facts,
    AnalysisKernelOutput,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisPayloadPartition, AnalysisPayloadPartitionPlan,
    AnalysisPayloadPlan, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_MANIFEST_MEDIA_TYPE,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

#[test]
fn chunk_conversion_stays_within_planned_requested_heap_and_preserves_names() {
    const PARTITION_BYTES: usize = 512 * 1024;
    const CHUNK_BYTES: usize = 64 * 1024;
    for count in [16_u64, 64] {
        // Caller-owned payloads and their borrowed inventory are outside the measured growth.
        let payloads = (0..count)
            .map(|ordinal| vec![ordinal as u8; PARTITION_BYTES])
            .collect::<Vec<_>>();
        let partitions = payloads
            .iter()
            .enumerate()
            .map(|(ordinal, payload)| AnalysisPayloadPartition {
                logical_identity: Digest::sha256((ordinal as u64).to_le_bytes()),
                payload,
            })
            .collect::<Vec<_>>();
        let plan = AnalysisPayloadPlan::new(
            AnalysisNumericalObjectKind::SolverResult,
            CHUNK_BYTES,
            partitions
                .iter()
                .map(|partition| AnalysisPayloadPartitionPlan {
                    logical_identity: partition.logical_identity,
                    encoded_bytes_upper_bound: partition.payload.len() as u64,
                })
                .collect(),
        )
        .unwrap();
        let facts = projected_chunked_output_resource_facts(&plan).unwrap();
        let root_identity = Digest::sha256(b"conversion-memory");
        let namespace = ObjectNamespace::ResultValue;
        let baseline = counting::begin_measurement();
        let prepared = prepare_partitioned_analysis_objects(
            AnalysisNumericalObjectKind::SolverResult,
            root_identity,
            &partitions,
            CHUNK_BYTES,
            namespace,
            ObjectInventoryLimits::default(),
        )
        .and_then(|prepared| {
            AnalysisKernelOutput::prepared(prepared)
                .into_prepared(CHUNK_BYTES, ObjectInventoryLimits::default())
        });
        let growth = counting::end_measurement().saturating_sub(baseline);
        let prepared = prepared.unwrap();
        let reserved = facts.construction_resident_bytes_admission_budget.0;
        assert!(
            growth <= reserved,
            "partitions={count}, peak={growth}, reservation={reserved}"
        );
        let closure_bytes = prepared
            .objects
            .iter()
            .map(|object| object.bytes.len() as u64)
            .sum::<u64>();
        assert!(closure_bytes <= facts.transport_closure_bytes_upper_bound.0);
        assert_eq!(
            prepared.manifest.encoded_length,
            count * PARTITION_BYTES as u64
        );
        assert_eq!(prepared.root, prepared.objects[0].descriptor);
        assert_eq!(prepared.root.namespace, namespace);
        assert_eq!(prepared.root.media_type, ANALYSIS_MANIFEST_MEDIA_TYPE);
        assert_eq!(
            prepared.root.logical_name,
            format!("analysis/manifests/{}", hex(prepared.root.digest))
        );
        assert_eq!(
            prepared.objects[0].bytes,
            prepared.manifest.canonical_bytes().unwrap()
        );
        assert_eq!(prepared.objects.len(), prepared.manifest.chunks.len() + 1);
        for (ordinal, (object, chunk)) in prepared.objects[1..]
            .iter()
            .zip(&prepared.manifest.chunks)
            .enumerate()
        {
            assert_eq!(object.descriptor.namespace, namespace);
            assert_eq!(object.descriptor.media_type, ANALYSIS_CHUNK_MEDIA_TYPE);
            assert_eq!(object.descriptor.digest, chunk.digest);
            assert_eq!(object.bytes.len() as u64, chunk.encoded_length);
            assert_eq!(
                object.descriptor.logical_name,
                format!("analysis/chunks/{ordinal:08}/{}", hex(chunk.digest))
            );
            object.validate().unwrap();
        }
        println!("partitions={count} peak_growth={growth} reserved={reserved} closure_bytes={closure_bytes}");
    }
}

fn hex(digest: Digest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
