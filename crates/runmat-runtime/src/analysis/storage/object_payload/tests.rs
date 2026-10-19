use super::*;
use runmat_analysis_fea::execution::{
    chunk_partitioned_numerical_payload, AnalysisNumericalObjectKind, AnalysisPayloadPartition,
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn selected_history_partition_reads_only_its_chunks_and_rejects_corruption() {
    let identities = [
        Digest::sha256(b"manifest"),
        Digest::sha256(b"first sample"),
        Digest::sha256(b"second sample"),
    ];
    let payloads = [vec![1; 1200], vec![2; 1700], vec![3; 1900]];
    let partitions: Vec<_> = identities
        .iter()
        .zip(&payloads)
        .map(|(identity, payload)| AnalysisPayloadPartition {
            logical_identity: *identity,
            payload,
        })
        .collect();
    let chunked = chunk_partitioned_numerical_payload(
        AnalysisNumericalObjectKind::History,
        identities[0],
        &partitions,
        256,
    )
    .unwrap();
    let objects: BTreeMap<_, _> = chunked
        .manifest
        .chunks
        .iter()
        .zip(&chunked.chunks)
        .map(|(descriptor, bytes)| (descriptor.digest, bytes.clone()))
        .collect();
    let selected = &chunked.manifest.partitions[1];
    let start = selected.first_chunk_ordinal as usize;
    let end = start + selected.chunk_count as usize;
    let expected: BTreeSet<_> = chunked.manifest.chunks[start..end]
        .iter()
        .map(|chunk| chunk.digest)
        .collect();
    let mut read = BTreeSet::new();
    let loaded = read_payload_partition(&chunked.manifest, identities[1], &mut |digest| {
        assert!(
            expected.contains(&digest),
            "must not read other history samples"
        );
        read.insert(digest);
        Ok(objects.get(&digest).cloned())
    })
    .unwrap();
    assert_eq!(loaded, payloads[1]);
    assert_eq!(read, expected);
    assert!(read_payload_partition(&chunked.manifest, identities[1], &mut |_| Ok(None)).is_err());
    assert!(
        read_payload_partition(&chunked.manifest, identities[1], &mut |digest| {
            let mut bytes = objects[&digest].clone();
            bytes[0] ^= 1;
            Ok(Some(bytes))
        })
        .is_err()
    );
    assert!(read_payload_partition(
        &chunked.manifest,
        Digest::sha256(b"absent"),
        &mut |_| panic!("unknown partition must not perform I/O")
    )
    .is_err());
}
