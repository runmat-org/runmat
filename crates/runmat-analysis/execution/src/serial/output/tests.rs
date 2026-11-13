use super::*;

fn prepared() -> PreparedAnalysisObjects {
    crate::prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::Field,
        Digest::sha256(b"field"),
        &[
            AnalysisPayloadPartition {
                logical_identity: Digest::sha256(b"a"),
                payload: b"abcdefg",
            },
            AnalysisPayloadPartition {
                logical_identity: Digest::sha256(b"b"),
                payload: b"hijklm",
            },
        ],
        4,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn refresh_root(objects: &mut PreparedAnalysisObjects) {
    let root = &mut objects.objects[0];
    root.bytes = objects.manifest.canonical_bytes().unwrap();
    root.descriptor.digest = Digest::sha256(&root.bytes);
    root.descriptor.encoded_length = root.bytes.len() as u64;
    let suffix = root
        .descriptor
        .digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    root.descriptor.logical_name = format!("analysis/manifests/{suffix}");
    objects.root = root.descriptor.clone();
}

#[test]
fn prepared_handoff_moves_every_buffer_without_gathering_or_rechunking() {
    let objects = prepared();
    let pointers = objects
        .objects
        .iter()
        .map(|o| o.bytes.as_ptr())
        .collect::<Vec<_>>();
    let output = AnalysisKernelOutput::prepared(objects);
    assert!(output.whole_payload().is_err());
    let retained = output
        .into_prepared(4, ObjectInventoryLimits::default())
        .unwrap();
    assert_eq!(
        pointers,
        retained
            .objects
            .iter()
            .map(|o| o.bytes.as_ptr())
            .collect::<Vec<_>>()
    );
    assert_eq!(retained.payload(), b"abcdefghijklm");
}

#[test]
fn prepared_handoff_rejects_substitutions_and_incomplete_closures() {
    let original = prepared();
    let mut cases = Vec::new();
    let mut changed = original.clone();
    changed.objects[0].descriptor.logical_name = "unbounded-root-name".repeat(100);
    changed.root = changed.objects[0].descriptor.clone();
    cases.push(changed);
    let mut changed = original.clone();
    changed.objects[1].descriptor.logical_name = "unbounded-chunk-name".repeat(100);
    cases.push(changed);
    let mut changed = original.clone();
    changed.objects.pop();
    cases.push(changed);
    let mut changed = original.clone();
    changed.objects[1].bytes[0] ^= 1;
    cases.push(changed);
    let mut changed = original.clone();
    changed.root.digest = Digest::sha256(b"wrong");
    cases.push(changed);
    let mut changed = original.clone();
    changed.objects[1].descriptor.namespace = ObjectNamespace::InputValue;
    cases.push(changed);
    let mut changed = original.clone();
    changed.objects[1].descriptor.media_type = "text/plain".into();
    cases.push(changed);
    let mut changed = original.clone();
    changed.manifest.payload_digest = Digest::sha256(b"wrong");
    refresh_root(&mut changed);
    cases.push(changed);
    let mut changed = original.clone();
    changed.manifest.partitions[0].payload_digest = Digest::sha256(b"wrong");
    refresh_root(&mut changed);
    cases.push(changed);
    let mut changed = original.clone();
    changed.manifest.chunks[0].digest = Digest::sha256(b"wrong");
    refresh_root(&mut changed);
    cases.push(changed);
    for changed in cases {
        assert!(AnalysisKernelOutput::prepared(changed)
            .into_prepared(4, ObjectInventoryLimits::default())
            .is_err());
    }
    let mut changed = AnalysisKernelOutput::prepared(original);
    changed.logical_identity = Digest::sha256(b"wrong");
    assert!(changed
        .into_prepared(4, ObjectInventoryLimits::default())
        .is_err());
}

#[test]
fn handoff_enforces_effective_policy_and_inventory_bounds() {
    assert_eq!(
        AnalysisOutputPolicy::new(usize::MAX, ObjectInventoryLimits::default())
            .unwrap()
            .chunk_bytes(),
        DEFAULT_ANALYSIS_CHUNK_BYTES
    );
    assert!(AnalysisOutputPolicy::new(0, ObjectInventoryLimits::default()).is_err());
    for chunk_bytes in [3, 5] {
        assert!(AnalysisKernelOutput::prepared(prepared())
            .into_prepared(chunk_bytes, ObjectInventoryLimits::default())
            .is_err());
    }
    for limits in [
        ObjectInventoryLimits {
            max_objects: 1,
            ..ObjectInventoryLimits::default()
        },
        ObjectInventoryLimits {
            max_total_bytes: 1,
            ..ObjectInventoryLimits::default()
        },
        ObjectInventoryLimits {
            max_object_bytes: 1,
            ..ObjectInventoryLimits::default()
        },
    ] {
        assert!(AnalysisKernelOutput::prepared(prepared())
            .into_prepared(4, limits)
            .is_err());
    }
    assert!(AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::Field,
        Digest::sha256(b"empty"),
        Vec::new()
    )
    .into_prepared(4, ObjectInventoryLimits::default())
    .is_err());
    assert!(AnalysisKernelOutput::partitioned(
        AnalysisNumericalObjectKind::Field,
        Digest::sha256(b"empty"),
        Vec::new()
    )
    .into_prepared(4, ObjectInventoryLimits::default())
    .is_err());
}
