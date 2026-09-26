use super::*;

fn builder(chunk_bytes: usize) -> AnalysisPayloadBuilder {
    AnalysisPayloadBuilder::new(
        AnalysisNumericalObjectKind::SolverResult,
        Digest::sha256(b"root"),
        chunk_bytes,
    )
    .unwrap()
}

#[test]
fn incremental_chunks_match_independently_described_partition_boundaries() {
    let first = Digest::sha256(b"first");
    let second = Digest::sha256(b"second");
    let mut incremental = builder(3);
    let mut input = b"abcde".to_vec();
    incremental
        .push(AnalysisPayloadPartition {
            logical_identity: first,
            payload: &input,
        })
        .unwrap();
    input.fill(0); // Successful push retained no reference to its source.
    incremental
        .push(AnalysisPayloadPartition {
            logical_identity: second,
            payload: b"fg",
        })
        .unwrap();
    let actual = incremental.finish().unwrap();
    let expected_chunks = vec![b"abc".to_vec(), b"de".to_vec(), b"fg".to_vec()];
    let expected = AnalysisObjectManifest {
        schema_version: 1,
        kind: AnalysisNumericalObjectKind::SolverResult,
        logical_identity: Digest::sha256(b"root"),
        payload_digest: Digest::sha256(b"abcdefg"),
        encoded_length: 7,
        chunks: vec![
            AnalysisChunkDescriptor {
                ordinal: 0,
                digest: Digest::sha256(b"abc"),
                encoded_length: 3,
            },
            AnalysisChunkDescriptor {
                ordinal: 1,
                digest: Digest::sha256(b"de"),
                encoded_length: 2,
            },
            AnalysisChunkDescriptor {
                ordinal: 2,
                digest: Digest::sha256(b"fg"),
                encoded_length: 2,
            },
        ],
        partitions: vec![
            AnalysisPayloadPartitionDescriptor {
                ordinal: 0,
                logical_identity: first,
                payload_digest: Digest::sha256(b"abcde"),
                encoded_length: 5,
                first_chunk_ordinal: 0,
                chunk_count: 2,
            },
            AnalysisPayloadPartitionDescriptor {
                ordinal: 1,
                logical_identity: second,
                payload_digest: Digest::sha256(b"fg"),
                encoded_length: 2,
                first_chunk_ordinal: 2,
                chunk_count: 1,
            },
        ],
    };
    assert_eq!(actual.chunks, expected_chunks);
    assert_eq!(actual.manifest, expected);
    assert_eq!(
        actual.manifest.canonical_bytes().unwrap(),
        expected.canonical_bytes().unwrap()
    );
    actual.revalidate(3).unwrap();
    assert!(actual.revalidate(2).is_err());
}

#[test]
fn bulk_and_incremental_paths_share_identical_canonical_output() {
    let partitions = [
        AnalysisPayloadPartition {
            logical_identity: Digest::sha256(b"a"),
            payload: b"12345",
        },
        AnalysisPayloadPartition {
            logical_identity: Digest::sha256(b"b"),
            payload: b"67890123",
        },
    ];
    for chunk_bytes in [1, 2, 3, 5, 8, usize::MAX] {
        let mut incremental = builder(chunk_bytes);
        for partition in partitions {
            incremental.push(partition).unwrap();
        }
        let actual = incremental.finish().unwrap();
        let bulk = chunk_partitioned_numerical_payload(
            AnalysisNumericalObjectKind::SolverResult,
            Digest::sha256(b"root"),
            &partitions,
            chunk_bytes,
        )
        .unwrap();
        assert_eq!(actual.chunks, bulk.chunks);
        assert_eq!(
            actual.manifest.canonical_bytes().unwrap(),
            bulk.manifest.canonical_bytes().unwrap()
        );
        actual.revalidate(chunk_bytes).unwrap();
    }
}

#[test]
fn rejected_partition_does_not_change_prior_output_or_reserve_identity() {
    assert!(
        AnalysisPayloadBuilder::new(AnalysisNumericalObjectKind::Field, Digest::sha256([]), 0)
            .is_err()
    );
    assert!(builder(1).finish().is_err());
    let first = AnalysisPayloadPartition {
        logical_identity: Digest::sha256(b"a"),
        payload: b"ok",
    };
    let mut actual = builder(1);
    actual.push(first).unwrap();
    assert!(actual.push(first).is_err());
    let next = Digest::sha256(b"b");
    assert!(actual
        .push(AnalysisPayloadPartition {
            logical_identity: next,
            payload: &[]
        })
        .is_err());
    let oversized = vec![0; MAX_CHUNKS + 1];
    assert!(actual
        .push(AnalysisPayloadPartition {
            logical_identity: next,
            payload: &oversized
        })
        .is_err());
    actual
        .push(AnalysisPayloadPartition {
            logical_identity: next,
            payload: b"yes",
        })
        .unwrap();
    let actual = actual.finish().unwrap();
    actual.revalidate(1).unwrap();
    assert_eq!(actual.manifest.encoded_length, 5);
    assert_eq!(actual.manifest.payload_digest, Digest::sha256(b"okyes"));
    assert_eq!(actual.manifest.partitions.len(), 2);
}

#[test]
fn append_arithmetic_checks_chunk_partition_and_byte_bounds_without_allocating() {
    assert_eq!(
        checked_append(MAX_CHUNKS - 1, MAX_CHUNKS - 1, 0, 1, 1).unwrap(),
        (1, 1)
    );
    assert!(checked_append(MAX_CHUNKS, 0, 0, 1, 1).is_err());
    assert!(checked_append(0, MAX_CHUNKS, 0, 1, 1).is_err());
    assert!(checked_append(usize::MAX, 0, 0, 1, 1).is_err());
    assert!(checked_append(0, usize::MAX, 0, 1, 1).is_err());
    assert!(checked_append(0, 0, u64::MAX, 1, 1).is_err());
    assert_eq!(checked_append(0, 0, 0, 7, usize::MAX).unwrap(), (1, 7));
}
