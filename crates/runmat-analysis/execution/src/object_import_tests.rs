use super::*;
use runmat_execution::value::ValueRef;

fn fixture() -> PreparedAnalysisObjects {
    prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::Field,
        digest("field"),
        &[
            runmat_analysis_fea::execution::AnalysisPayloadPartition {
                logical_identity: digest("first"),
                payload: b"abcdefghijk",
            },
            runmat_analysis_fea::execution::AnalysisPayloadPartition {
                logical_identity: digest("second"),
                payload: b"lmnopqrstuv",
            },
        ],
        4,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn store_and_root(prepared: &PreparedAnalysisObjects) -> (RecordingObjects, ValueRef) {
    let bytes = prepared.manifest.canonical_bytes().unwrap();
    let hash = Digest::sha256(&bytes);
    let mut root = prepared
        .root_reference(&access(), ValueRefKind::ResultObject)
        .unwrap();
    root.logical_digest = hash;
    root.id = access().value_id(hash);
    root.encoded_length = bytes.len() as u64;
    let mut objects = prepared
        .objects
        .iter()
        .skip(1)
        .map(|object| (object.descriptor.digest, object.bytes.clone()))
        .collect::<BTreeMap<_, _>>();
    objects.insert(hash, bytes);
    (
        RecordingObjects {
            objects,
            reads: RefCell::new(Vec::new()),
        },
        root,
    )
}

#[test]
fn full_and_partial_import_reject_semantic_digest_substitution() {
    for substitute_partition in [false, true] {
        let mut prepared = fixture();
        if substitute_partition {
            prepared.manifest.partitions[0].payload_digest = digest("wrong-partition");
        } else {
            prepared.manifest.payload_digest = digest("wrong-total");
        }
        let (store, root) = store_and_root(&prepared);
        assert!(import_analysis_objects(
            &store,
            &root,
            AnalysisNumericalObjectKind::Field,
            digest("field"),
            prepared.manifest.payload_digest,
            &access(),
            ObjectInventoryLimits::default(),
        )
        .is_err());
        let partial = import_analysis_partition(
            &store,
            AnalysisPartitionImport {
                root: &root,
                expected_kind: AnalysisNumericalObjectKind::Field,
                expected_logical_identity: digest("field"),
                expected_payload_digest: prepared.manifest.payload_digest,
                partition_identity: digest("first"),
                access: &access(),
                limits: ObjectInventoryLimits::default(),
            },
        );
        // Partial access proves the selected semantic digest, not unread complete content.
        assert_eq!(partial.is_err(), substitute_partition);
    }
}

#[test]
fn imported_partition_reads_exactly_its_manifest_and_chunks() {
    let prepared = fixture();
    let (store, root) = store_and_root(&prepared);
    let selected = &prepared.manifest.partitions[1];
    let imported = import_analysis_partition(
        &store,
        AnalysisPartitionImport {
            root: &root,
            expected_kind: AnalysisNumericalObjectKind::Field,
            expected_logical_identity: digest("field"),
            expected_payload_digest: prepared.manifest.payload_digest,
            partition_identity: digest("second"),
            access: &access(),
            limits: ObjectInventoryLimits::default(),
        },
    )
    .unwrap();
    assert_eq!(imported.payload(), b"lmnopqrstuv");
    let start = selected.first_chunk_ordinal as usize;
    let end = start + selected.chunk_count as usize;
    let expected = std::iter::once(root.logical_digest)
        .chain(
            prepared.manifest.chunks[start..end]
                .iter()
                .map(|chunk| chunk.digest),
        )
        .collect::<Vec<_>>();
    assert_eq!(*store.reads.borrow(), expected);
}

#[test]
fn full_import_preserves_payload_and_rejects_short_verified_read() {
    let prepared = fixture();
    let (mut store, root) = store_and_root(&prepared);
    let imported = import_analysis_objects(
        &store,
        &root,
        AnalysisNumericalObjectKind::Field,
        digest("field"),
        prepared.manifest.payload_digest,
        &access(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported.payload(), b"abcdefghijklmnopqrstuv");
    store
        .objects
        .get_mut(&prepared.manifest.chunks[0].digest)
        .unwrap()
        .pop();
    assert!(import_analysis_objects(
        &store,
        &root,
        AnalysisNumericalObjectKind::Field,
        digest("field"),
        prepared.manifest.payload_digest,
        &access(),
        ObjectInventoryLimits::default(),
    )
    .is_err());
}
