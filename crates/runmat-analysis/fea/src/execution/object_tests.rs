use super::*;

fn prepared(chunk_bytes: usize) -> ChunkedAnalysisPayload {
    chunk_partitioned_numerical_payload(
        AnalysisNumericalObjectKind::SolverResult,
        Digest::sha256(b"result"),
        &[
            AnalysisPayloadPartition {
                logical_identity: Digest::sha256(b"metadata"),
                payload: b"abc",
            },
            AnalysisPayloadPartition {
                logical_identity: Digest::sha256(b"sample"),
                payload: b"defghijklmnop",
            },
        ],
        chunk_bytes,
    )
    .unwrap()
}

#[test]
fn whole_payload_partition_binds_the_complete_object_and_rejects_substitution() {
    let object = chunk_numerical_payload(
        AnalysisNumericalObjectKind::SeriesReductionResult,
        Digest::sha256(b"series"),
        b"accepted series payload",
        4,
    )
    .unwrap();
    object.manifest.whole_payload_partition().unwrap();
    let restored =
        AnalysisObjectManifest::from_canonical_bytes(&object.manifest.canonical_bytes().unwrap())
            .unwrap();
    restored.whole_payload_partition().unwrap();
    for substitute_identity in [true, false] {
        let mut substituted = restored.clone();
        if substitute_identity {
            substituted.partitions[0].logical_identity = Digest::sha256(b"foreign series");
        } else {
            substituted.partitions[0].payload_digest = Digest::sha256(b"foreign payload");
        }
        // Structurally valid descriptors must not be mistaken for the whole object role.
        substituted.validate().unwrap();
        assert!(substituted.whole_payload_partition().is_err());
    }
    assert!(prepared(4).manifest.whole_payload_partition().is_err());
}

#[test]
fn incremental_digest_matches_concatenation_across_chunk_boundaries() {
    for chunk_bytes in [1, 2, 3, 4, 5, 12, 16, usize::MAX] {
        let object = prepared(chunk_bytes);
        object.revalidate(chunk_bytes).unwrap();
        assert_eq!(
            object.manifest.payload_digest,
            Digest::sha256(b"abcdefghijklmnop")
        );
        assert_eq!(object.manifest.encoded_length, 16);
        assert_eq!(
            object.manifest.partitions[0].payload_digest,
            Digest::sha256(b"abc")
        );
        assert_eq!(
            object.manifest.partitions[1].payload_digest,
            Digest::sha256(b"defghijklmnop")
        );
        let encoded = object.manifest.canonical_bytes().unwrap();
        assert_eq!(
            AnalysisObjectManifest::from_canonical_bytes(&encoded).unwrap(),
            object.manifest
        );
    }
}

#[test]
fn closure_validation_rejects_corruption_without_gathering_or_index_panics() {
    let original = prepared(4);
    let mut cases = Vec::new();
    let mut object = original.clone();
    object.chunks.pop();
    cases.push(object);
    let mut object = original.clone();
    object.chunks.push(vec![1]);
    cases.push(object);
    let mut object = original.clone();
    object.chunks[1][0] ^= 1;
    cases.push(object);
    let mut object = original.clone();
    object.manifest.partitions[0].first_chunk_ordinal = u32::MAX;
    cases.push(object);
    let mut object = original.clone();
    object.manifest.partitions[1].chunk_count = u32::MAX;
    cases.push(object);
    let mut object = original.clone();
    object.manifest.partitions[1].payload_digest = Digest::sha256(b"substituted");
    cases.push(object);
    let mut object = original.clone();
    object.manifest.payload_digest = Digest::sha256(b"substituted");
    cases.push(object);
    let mut object = original.clone();
    object.manifest.schema_version = 2;
    cases.push(object);
    let mut object = original.clone();
    object.chunks[0].clear();
    cases.push(object);
    let mut object = original.clone();
    object.manifest.partitions[1].logical_identity = object.manifest.partitions[0].logical_identity;
    cases.push(object);
    for object in cases {
        assert!(object.revalidate(4).is_err());
    }
    assert!(original.revalidate(0).is_err());
    assert!(original.revalidate(3).is_err());
    assert!(original.revalidate(5).is_err());
}

#[test]
fn valid_hashes_do_not_admit_noncanonical_chunk_lengths() {
    let mut object = prepared(4);
    // Preserve complete content, all lengths and hashes, but shift one boundary.
    let byte = object.chunks[1].pop().unwrap();
    object.chunks[2].insert(0, byte);
    for (descriptor, bytes) in object.manifest.chunks.iter_mut().zip(&object.chunks) {
        descriptor.encoded_length = bytes.len() as u64;
        descriptor.digest = Digest::sha256(bytes);
    }
    object.manifest.validate().unwrap();
    assert!(object.revalidate(4).is_err());
}

#[test]
fn manifest_validation_uses_checked_borrowed_slots_without_reordering() {
    let original = prepared(4).manifest;
    let mut manifest = original.clone();
    // Logical identities need not be sorted in canonical semantic order.
    manifest.partitions[0].logical_identity = Digest::from_bytes([255; 32]);
    manifest.partitions[1].logical_identity = Digest::from_bytes([0; 32]);
    let before = manifest.clone();
    manifest.validate().unwrap();
    assert_eq!(manifest, before);
    assert_eq!(
        AnalysisObjectManifest::validation_scratch_bytes(0).unwrap(),
        0
    );
    assert_eq!(
        AnalysisObjectManifest::validation_scratch_bytes(2).unwrap(),
        (2 * std::mem::size_of::<&Digest>()) as u64
    );
    assert!(AnalysisObjectManifest::validation_scratch_bytes(usize::MAX).is_err());
    manifest.partitions[1].logical_identity = manifest.partitions[0].logical_identity;
    assert!(matches!(
        manifest.validate(),
        Err(AnalysisExecutionError::Invalid(_))
    ));
}

#[test]
fn manifest_validation_rejects_nonadjacent_duplicate_identities() {
    let plan = AnalysisPayloadPlan::new(
        AnalysisNumericalObjectKind::SolverResult,
        4,
        [3, 1, 2]
            .into_iter()
            .map(|id| AnalysisPayloadPartitionPlan {
                logical_identity: Digest::from_bytes([id; 32]),
                encoded_bytes_upper_bound: 4,
            })
            .collect(),
    )
    .unwrap();
    let mut builder = AnalysisPayloadBuilder::new(Digest::sha256(b"result"), plan).unwrap();
    for id in [3, 1, 2] {
        builder
            .push(AnalysisPayloadPartition {
                logical_identity: Digest::from_bytes([id; 32]),
                payload: b"data",
            })
            .unwrap();
    }
    let mut manifest = builder.finish().unwrap().manifest;
    manifest.validate().unwrap();
    manifest.partitions[2].logical_identity = manifest.partitions[0].logical_identity;
    assert!(matches!(
        manifest.validate(),
        Err(AnalysisExecutionError::Invalid(_))
    ));
}
