use super::*;

#[test]
fn projection_counts_each_partition_and_bounds_actual_root_wire() {
    for chunk_bytes in [1, 7, 1024] {
        let data = [vec![1; 15], vec![2; 15], vec![3; 1]];
        let mut profiler = AnalysisTransportProfiler::new(chunk_bytes).unwrap();
        let plan = AnalysisPayloadPlan::new(
            AnalysisNumericalObjectKind::SolverResult,
            chunk_bytes,
            data.iter()
                .map(|payload| AnalysisPayloadPartitionPlan {
                    logical_identity: Digest::sha256(payload),
                    encoded_bytes_upper_bound: payload.len() as u64,
                })
                .collect(),
        )
        .unwrap();
        let planned_transport = *plan.transport_profile();
        let mut builder = AnalysisPayloadBuilder::new(Digest::sha256(b"result"), plan).unwrap();
        for payload in &data {
            profiler.push(payload.len() as u64).unwrap();
            builder
                .push(AnalysisPayloadPartition {
                    logical_identity: Digest::sha256(payload),
                    payload,
                })
                .unwrap();
        }
        let profile = profiler
            .finish(AnalysisNumericalObjectKind::SolverResult)
            .unwrap();
        assert_eq!(planned_transport, profile);
        let actual = builder.finish().unwrap();
        assert_eq!(profile.partition_count, 3);
        assert_eq!(
            profile.manifest_validation_scratch_bytes,
            AnalysisObjectManifest::validation_scratch_bytes(3).unwrap()
        );
        assert_eq!(
            profile.chunk_count_upper_bound as usize,
            actual.chunks.len()
        );
        assert_eq!(profile.payload_bytes_upper_bound, 31);
        assert_eq!(
            profile.maximum_partition_chunks_upper_bound,
            15_u64.div_ceil(chunk_bytes as u64)
        );
        let root = actual.manifest.canonical_bytes().unwrap();
        assert!(root.len() as u64 <= profile.root.encoded_bytes_upper_bound);
        assert!(root.len() as u64 + 31 <= profile.total_bytes_upper_bound);
        assert!(actual
            .chunks
            .iter()
            .all(|chunk| chunk.len() as u64 <= profile.largest_chunk_bytes_upper_bound));
        let actual_profile =
            runmat_canonical_codec::profile_serialized_shape(&actual.manifest, MANIFEST_LIMITS)
                .unwrap();
        assert_eq!(actual_profile, profile.root);

        // Independent legacy shape: extracting the serializer must not change manifest bytes.
        let m = &actual.manifest;
        let legacy = serde_json::json!({
            "schema_version": m.schema_version, "kind": m.kind,
            "logical_identity": m.logical_identity, "payload_digest": m.payload_digest,
            "encoded_length": m.encoded_length, "chunks": m.chunks, "partitions": m.partitions,
        });
        assert_eq!(
            root,
            encode_contract(MANIFEST_PREFIX, MANIFEST_DOMAIN, &legacy, MANIFEST_LIMITS).unwrap()
        );
    }
}

#[test]
fn builder_shape_rejects_insufficient_capacity_and_checked_overflow() {
    let mut profiler = AnalysisTransportProfiler::new(4).unwrap();
    profiler.push(9).unwrap();
    profiler.push(1).unwrap();
    let profile = profiler
        .finish(AnalysisNumericalObjectKind::SolverResult)
        .unwrap();
    assert!(profile.builder_resident_bytes(1).is_err());
    assert!(profile.builder_resident_bytes(usize::MAX).is_err());
    assert!(profile.builder_resident_bytes(2).is_ok());
    let mut overflow = profile;
    overflow.payload_bytes_upper_bound = u64::MAX;
    assert!(overflow.builder_resident_bytes(2).is_err());
    let mut slot_overflow = profile;
    slot_overflow.maximum_partition_chunks_upper_bound = u64::MAX;
    assert!(slot_overflow.builder_resident_bytes(2).is_err());
}

#[test]
fn partition_remainders_are_not_coalesced() {
    let mut profiler = AnalysisTransportProfiler::new(10).unwrap();
    for _ in 0..3 {
        profiler.push(11).unwrap();
    }
    let profile = profiler
        .finish(AnalysisNumericalObjectKind::SolverResult)
        .unwrap();
    assert_eq!(profile.chunk_count_upper_bound, 6);
    assert_eq!(profile.payload_bytes_upper_bound.div_ceil(10), 4);
}

#[test]
fn rejects_invalid_and_overflowing_chunks_without_chunk_allocations() {
    assert!(AnalysisTransportProfiler::new(0).is_err());
    assert!(AnalysisTransportProfiler::new(1)
        .unwrap()
        .finish(AnalysisNumericalObjectKind::SolverResult)
        .is_err());
    let mut profiler = AnalysisTransportProfiler::new(1).unwrap();
    assert!(profiler.push(0).is_err());
    assert!(profiler.push(MAX_CHUNKS as u64 + 1).is_err());
    profiler.push(1).unwrap();
    assert_eq!(
        profiler
            .finish(AnalysisNumericalObjectKind::SolverResult)
            .unwrap()
            .chunk_count_upper_bound,
        1
    );
    // The root manifest has no fixed byte ceiling: a manifest listing the maximum chunk count
    // profiles, and the run's admitted budget decides whether it fits.
    let mut full_root = AnalysisTransportProfiler::new(1).unwrap();
    full_root.push(MAX_CHUNKS as u64).unwrap();
    assert!(full_root
        .finish(AnalysisNumericalObjectKind::SolverResult)
        .is_ok());
    let mut overflow = AnalysisTransportProfiler::new(1).unwrap();
    overflow.payload_bytes = u64::MAX;
    assert!(overflow.push(1).is_err());
}
