use super::*;

fn entry(id: u8, bytes: u64) -> AnalysisPayloadPartitionPlan {
    AnalysisPayloadPartitionPlan {
        logical_identity: Digest::from_bytes([id; 32]),
        encoded_bytes_upper_bound: bytes,
    }
}

#[test]
fn plan_rejects_invalid_and_duplicate_inventories_before_builder_storage() {
    let kind = AnalysisNumericalObjectKind::SolverResult;
    assert!(AnalysisPayloadPlan::new(kind, 1, vec![]).is_err());
    assert!(AnalysisPayloadPlan::new(kind, 0, vec![entry(1, 4)]).is_err());
    assert!(AnalysisPayloadPlan::new(kind, 1, vec![entry(1, 0)]).is_err());
    assert!(
        AnalysisPayloadPlan::new(kind, 1, vec![entry(1, 4), entry(2, 2), entry(1, 1)]).is_err()
    );
    assert!(AnalysisPayloadPlan::new(kind, 1, vec![entry(1, u64::MAX)]).is_err());
    assert!(slots::<u64>(usize::MAX).is_err());
    assert!(reserved::<u64>(usize::MAX).is_err());
    assert!(AnalysisPayloadPlan::reserve_partitions(0).is_err());
    assert!(AnalysisPayloadPlan::reserve_partitions(MAX_CHUNKS + 1).is_err());
}

#[test]
fn plan_counts_owned_capacity_and_each_live_inventory() {
    let mut entries = Vec::with_capacity(10);
    entries.extend([entry(3, 10), entry(1, 5)]);
    let capacity = entries.capacity();
    let plan =
        AnalysisPayloadPlan::new(AnalysisNumericalObjectKind::SolverResult, 4, entries).unwrap();
    assert_eq!(plan.transport_profile().chunk_count_upper_bound, 5);
    assert_eq!(
        plan.transport_profile()
            .maximum_partition_chunks_upper_bound,
        3
    );
    assert_eq!(
        plan.resident_bytes().unwrap(),
        plan.transport_profile()
            .builder_resident_bytes(capacity)
            .unwrap()
    );
    assert_eq!(
        plan.resident_bytes().unwrap(),
        slots::<AnalysisPayloadPartitionPlan>(capacity).unwrap()
            + slots::<AnalysisChunkDescriptor>(5).unwrap()
            + slots::<Vec<u8>>(5).unwrap()
            + slots::<AnalysisPayloadPartitionDescriptor>(2).unwrap()
            + slots::<Vec<u8>>(3).unwrap()
            + 15
            + AnalysisObjectManifest::validation_scratch_bytes(2).unwrap()
    );
    // Moving the plan into the builder does not reorder authored semantic identities.
    let mut builder = AnalysisPayloadBuilder::new(Digest::sha256(b"root"), plan).unwrap();
    builder
        .push(AnalysisPayloadPartition {
            logical_identity: entry(3, 0).logical_identity,
            payload: b"small",
        })
        .unwrap();
    builder
        .push(AnalysisPayloadPartition {
            logical_identity: entry(1, 0).logical_identity,
            payload: b"tiny",
        })
        .unwrap();
    let output = builder.finish().unwrap();
    output.revalidate(4).unwrap();
    assert_eq!(
        output.manifest.partitions[0].logical_identity,
        entry(3, 0).logical_identity
    );
    assert_eq!(output.manifest.encoded_length, 9);
}
