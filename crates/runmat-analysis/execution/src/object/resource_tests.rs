use super::*;

#[test]
fn chunked_input_inventory_preserves_partition_cost_and_rejects_missing_bytes() {
    let limits = runmat_canonical_codec::CanonicalLimits::new(4096, 100, 256, 8);
    let payloads = [
        runmat_canonical_codec::encode_contract(b"left\0", "left", &[0.125; 17], limits).unwrap(),
        runmat_canonical_codec::encode_contract(b"right\0", "right", &"é𝄞", limits).unwrap(),
    ];
    let expected = payloads
        .iter()
        .map(|p| {
            runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(p).unwrap()
        })
        .sum::<u64>();
    for chunk_bytes in [1, 7, 1024] {
        let objects = prepare_partitioned_analysis_objects(
            AnalysisNumericalObjectKind::SolverInvocation,
            Digest::sha256("partitioned-original"),
            &[
                AnalysisPayloadPartition {
                    logical_identity: Digest::sha256("left"),
                    payload: &payloads[0],
                },
                AnalysisPayloadPartition {
                    logical_identity: Digest::sha256("right"),
                    payload: &payloads[1],
                },
            ],
            chunk_bytes,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let access = ArtifactAccess {
            authorization_scope: "partition-reference-schema".into(),
            encryption_context: Digest::sha256("partition-reference-key"),
        };
        let root = objects
            .root_reference(&access, ValueRefKind::ResultObject)
            .unwrap();
        let references = analysis_object_references(
            &objects.manifest,
            &root,
            &access,
            ValueRefKind::ResultObject,
        )
        .unwrap();
        assert_eq!(
            references,
            objects
                .unique_references(&access, ValueRefKind::ResultObject)
                .unwrap()
        );
        for reference in references {
            assert_eq!(reference.schema_version, VALUE_PAYLOAD_SCHEMA_VERSION);
            reference
                .validate(runmat_execution::value::ValueLimits::default())
                .unwrap();
        }
        let facts = prepared_input_resource_facts_from_refs(std::iter::once(&objects)).unwrap();
        assert_eq!(facts.decoded_resident_bytes_admission_budget.0, expected);
        assert_eq!(
            facts.encoded_payload_bytes.0,
            payloads.iter().map(|p| p.len() as u64).sum::<u64>()
        );
        let mut missing = objects.clone();
        missing.objects.pop();
        assert!(prepared_input_resource_facts_from_refs(std::iter::once(&missing)).is_err());
        let mut changed = objects.clone();
        changed.objects[1].bytes[0] ^= 1;
        assert!(prepared_input_resource_facts_from_refs(std::iter::once(&changed)).is_err());
        let mut invalid_range = objects;
        invalid_range.manifest.partitions[0].first_chunk_ordinal = u32::MAX;
        assert!(prepared_input_resource_facts_from_refs(std::iter::once(&invalid_range)).is_err());
    }
}

#[test]
fn rebinding_unpartitioned_identity_preserves_payload_chunks() {
    let limits = ObjectInventoryLimits::default();
    let payload = vec![7_u8; 129];
    let prepared = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
        Digest::sha256("scope-workload"),
        &payload,
        32,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let old_root = prepared.root.digest;
    let chunk_addresses = prepared
        .objects
        .iter()
        .skip(1)
        .map(|object| object.bytes.as_ptr())
        .collect::<Vec<_>>();
    let rebound = rebind_unpartitioned_analysis_identity(
        prepared,
        Digest::sha256("assembled-workload"),
        limits,
    )
    .unwrap();
    assert_ne!(rebound.root.digest, old_root);
    assert_eq!(
        rebound.manifest.logical_identity,
        Digest::sha256("assembled-workload")
    );
    assert_eq!(
        rebound.manifest.partitions[0].logical_identity,
        rebound.manifest.logical_identity
    );
    assert_eq!(rebound.payload(), payload);
    assert_eq!(
        rebound
            .objects
            .iter()
            .skip(1)
            .map(|object| object.bytes.as_ptr())
            .collect::<Vec<_>>(),
        chunk_addresses
    );
    assert_eq!(
        rebound,
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::StructuralTransientWorkload,
            Digest::sha256("assembled-workload"),
            &payload,
            32,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap()
    );

    let partitioned = prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
        Digest::sha256("partitioned-workload"),
        &[
            AnalysisPayloadPartition {
                logical_identity: Digest::sha256("left"),
                payload: &[1_u8; 8],
            },
            AnalysisPayloadPartition {
                logical_identity: Digest::sha256("right"),
                payload: &[2_u8; 8],
            },
        ],
        32,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    assert!(rebind_unpartitioned_analysis_identity(
        partitioned,
        Digest::sha256("rebinding-must-fail"),
        limits,
    )
    .is_err());

    let mut substituted = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
        Digest::sha256("scope-workload"),
        &payload,
        32,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let original = &substituted.objects[1];
    substituted.objects[1] = LogicalObject::new(
        original.descriptor.namespace,
        original.descriptor.logical_name.clone(),
        original.descriptor.media_type.clone(),
        vec![9_u8; original.bytes.len()],
    )
    .unwrap();
    assert!(rebind_unpartitioned_analysis_identity(
        substituted,
        Digest::sha256("rebinding-must-fail"),
        limits,
    )
    .is_err());
}

#[test]
fn prepared_output_must_fit_semantic_and_transport_envelopes_independently() {
    let prepared = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationResult,
        Digest::sha256("output"),
        &[7_u8; 128],
        32,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let actual_closure = prepared
        .objects
        .iter()
        .map(|object| (object.descriptor.digest, object.bytes.len() as u64))
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_values()
        .sum::<u64>();
    let admitted = PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(128),
        transport_closure_bytes_upper_bound: ByteCount(actual_closure),
        construction_resident_bytes_admission_budget: ByteCount(128),
    };
    validate_prepared_output_resource_facts(&prepared, &admitted).unwrap();

    let mut understated_payload = admitted.clone();
    understated_payload.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(validate_prepared_output_resource_facts(&prepared, &understated_payload).is_err());

    let mut understated_transport = admitted;
    understated_transport.transport_closure_bytes_upper_bound.0 -= 1;
    assert!(validate_prepared_output_resource_facts(&prepared, &understated_transport).is_err());
}
