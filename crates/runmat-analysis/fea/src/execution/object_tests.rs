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
