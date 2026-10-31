use runmat_execution::identity::{Digest, NodeLeaseId, ValueId, WorkerId};
use runmat_execution::value::{
    DenseValue, ElementType, InlineValue, RegisteredData, RegisteredField, ResidentFence,
    ResidentValueAuthority, SparseValue, ValueLimits, ValuePayload, ValueRef, ValueRefKind,
    ValueTransportContext,
};

#[test]
fn nested_payloads_obey_depth_and_node_limits() {
    let payload = ValuePayload::Inline(Box::new(InlineValue::Cell {
        shape: vec![1],
        values: vec![ValuePayload::Inline(Box::new(InlineValue::Cell {
            shape: vec![1],
            values: vec![ValuePayload::Inline(Box::new(InlineValue::Logical(true)))],
        }))],
    }));
    assert!(payload
        .validate(ValueLimits {
            max_depth: 1,
            ..ValueLimits::default()
        })
        .is_err());
    assert!(payload.validate(ValueLimits::default()).is_ok());
}

#[test]
fn top_level_payloads_share_byte_element_and_text_budgets() {
    let dense = || {
        ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
            element_type: ElementType::U8,
            shape: vec![1],
            little_endian_data: vec![1],
        })))
    };
    let error = ValuePayload::validate_all(
        &[dense(), dense()],
        ValueLimits {
            max_inline_bytes: 1,
            ..ValueLimits::default()
        },
    )
    .expect_err("top-level payload bytes share one budget");
    assert!(matches!(
        error,
        runmat_execution::ContractError::Limit {
            field: "inline value bytes",
            limit: 1
        }
    ));

    let error = ValuePayload::validate_all(
        &[dense(), dense()],
        ValueLimits {
            max_elements: 1,
            ..ValueLimits::default()
        },
    )
    .expect_err("top-level payload elements share one budget");
    assert!(matches!(
        error,
        runmat_execution::ContractError::Limit {
            field: "value elements",
            limit: 1
        }
    ));

    let text = || ValuePayload::Inline(Box::new(InlineValue::String("four".into())));
    let error = ValuePayload::validate_all(
        &[text(), text()],
        ValueLimits {
            max_text_bytes: 7,
            ..ValueLimits::default()
        },
    )
    .expect_err("top-level payload text shares one budget");
    assert!(matches!(
        error,
        runmat_execution::ContractError::Limit {
            field: "value text bytes",
            limit: 7
        }
    ));
}

#[test]
fn durable_object_array_elements_count_as_nodes_and_depth() {
    use runmat_execution::value::{DurableObjectArray, DurablePlainObject};
    let object = DurablePlainObject {
        class_identity: runmat_types::ClassIdentity::new("PreferencePayload").unwrap(),
        properties: Vec::new(),
        dynamic_properties: Vec::new(),
        decomposition: None,
    };
    let payload = ValuePayload::Inline(Box::new(InlineValue::DurableObjectArray(
        DurableObjectArray {
            class_identity: object.class_identity.clone(),
            shape: vec![1, 2],
            elements: vec![object.clone(), object],
        },
    )));
    assert!(
        payload.validate(ValueLimits::default()).is_err(),
        "V2 rejects durable tags"
    );
    assert!(payload
        .validate_for_durable_storage(ValueLimits {
            max_nodes: 2,
            ..ValueLimits::default()
        })
        .is_err());
    assert!(payload
        .validate_for_durable_storage(ValueLimits {
            max_depth: 0,
            ..ValueLimits::default()
        })
        .is_err());
    assert!(payload
        .validate_for_durable_storage(ValueLimits::default())
        .is_ok());
    let null = ValuePayload::Inline(Box::new(InlineValue::Null));
    assert!(
        null.validate(ValueLimits::default()).is_ok(),
        "V2 wire null remains defined"
    );
    assert!(null
        .validate_for_durable_storage(ValueLimits::default())
        .is_err());
}

#[test]
fn struct_fields_must_be_sorted_and_unique() {
    use runmat_execution::value::StructField;
    let scalar = || ValuePayload::Inline(Box::new(InlineValue::Logical(true)));
    let payload = ValuePayload::Inline(Box::new(InlineValue::Struct(vec![
        StructField {
            name: "z".into(),
            value: scalar(),
        },
        StructField {
            name: "a".into(),
            value: scalar(),
        },
    ])));
    assert!(payload.validate(ValueLimits::default()).is_err());
}

#[test]
fn dense_and_sparse_storage_must_match_declared_shapes() {
    let dense = ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::F64,
        shape: vec![2, 2],
        little_endian_data: vec![0; 24],
    })));
    assert!(dense.validate(ValueLimits::default()).is_err());

    let sparse = ValuePayload::Inline(Box::new(InlineValue::Sparse(SparseValue {
        element_type: ElementType::F64,
        rows: 2,
        columns: 2,
        column_offsets: vec![0, 2, 1],
        row_indices: vec![0],
        little_endian_data: vec![0; 8],
    })));
    assert!(sparse.validate(ValueLimits::default()).is_err());
}

#[test]
fn registered_fields_are_validated_without_projecting_live_values() {
    let scalar = || ValuePayload::Inline(Box::new(InlineValue::Logical(true)));
    let payload = ValuePayload::Inline(Box::new(InlineValue::Symbolic(RegisteredData {
        type_identity: "symbolic/v1".into(),
        schema_version: 1,
        fields: vec![
            RegisteredField {
                name: "z".into(),
                value: scalar(),
            },
            RegisteredField {
                name: "a".into(),
                value: scalar(),
            },
        ],
    })));
    assert!(payload.validate(ValueLimits::default()).is_err());
}

#[test]
fn resident_values_require_and_bind_a_worker_fence() {
    let mut reference = ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: ValueId::derive(&[b"value"]),
        logical_digest: Digest::sha256(b"value"),
        encoded_length: 1,
        media_type: "application/runmat-value".into(),
        value_schema: "runmat-value/v2".into(),
        encryption_context: Digest::sha256(b"context"),
        kind: ValueRefKind::ResidentObject,
        authorization_scope: "scope".into(),
        resident_fence: None,
    };
    assert!(ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .is_err());

    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"attempt"]);
    let device_id = runmat_execution::resource::AcceleratorDeviceId::new("gpu-0").unwrap();
    let inventory_epoch = 1;
    let fencing_token = 2;
    let worker_id = WorkerId::derive(&[b"worker"]);
    let node_lease_id = NodeLeaseId::derive(&[b"lease"]);
    let accelerator_lease = runmat_execution::resource::AcceleratorDeviceLease::for_attempt(
        attempt_id,
        fencing_token,
        &runmat_execution::resource::AcceleratorDevice {
            id: device_id.clone(),
            allocation_domain: runmat_execution::resource::AcceleratorAllocationDomainId::new(
                "allocation:test",
            )
            .unwrap(),
            class: runmat_execution::resource::AcceleratorClass::new("gpu").unwrap(),
            provider: runmat_execution::resource::AcceleratorProvider {
                id: runmat_execution::resource::AcceleratorProviderId::new("wgpu").unwrap(),
                version: runmat_execution::resource::AcceleratorProviderVersion::new("1").unwrap(),
                abi_fingerprint: Digest::sha256(b"provider"),
            },
            max_allocation_bytes: 1024,
            features: std::collections::BTreeSet::from([
                runmat_execution::resource::AcceleratorFeature::Compute,
            ]),
            inventory_epoch,
        },
    )
    .unwrap();
    reference.resident_fence = Some(ResidentFence {
        worker_id,
        node_lease_id,
        process_generation: 2,
        accelerator: Some(runmat_execution::value::ResidentAcceleratorFence {
            attempt_id,
            lease: accelerator_lease.clone(),
        }),
    });
    let resident = ValuePayload::Object(Box::new(reference.clone()));
    assert!(resident.validate(ValueLimits::default()).is_ok());
    assert!(resident
        .validate_for_transport(ValueLimits::default(), ValueTransportContext::Portable)
        .is_err());
    let accelerator_leases = [accelerator_lease];
    assert!(resident
        .validate_for_transport(
            ValueLimits::default(),
            ValueTransportContext::Resident(ResidentValueAuthority {
                worker_id,
                node_lease_id,
                process_generation: 2,
                attempt_id,
                accelerator_leases: &accelerator_leases,
            }),
        )
        .is_ok());
    assert!(resident
        .validate_for_transport(
            ValueLimits::default(),
            ValueTransportContext::Resident(ResidentValueAuthority {
                worker_id,
                node_lease_id,
                process_generation: 3,
                attempt_id,
                accelerator_leases: &accelerator_leases,
            }),
        )
        .is_err());

    reference.kind = ValueRefKind::ResultObject;
    assert!(ValuePayload::Object(Box::new(reference))
        .validate(ValueLimits::default())
        .is_err());
}

#[test]
fn logical_identity_is_independent_of_inline_or_object_placement() {
    let inline = ValuePayload::Inline(Box::new(InlineValue::String("payload".into())));
    let logical_digest = inline.logical_digest().unwrap();
    let reference = ValuePayload::Object(Box::new(ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: ValueId::derive(&[b"value"]),
        logical_digest,
        encoded_length: 1024,
        media_type: "application/runmat-value".into(),
        value_schema: "runmat-value/v2".into(),
        encryption_context: Digest::sha256(b"context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: "scope".into(),
        resident_fence: None,
    }));
    assert_eq!(reference.logical_digest().unwrap(), logical_digest);

    let nested_inline = ValuePayload::Inline(Box::new(InlineValue::Cell {
        shape: vec![1, 1],
        values: vec![inline],
    }));
    let nested_reference = ValuePayload::Inline(Box::new(InlineValue::Cell {
        shape: vec![1, 1],
        values: vec![reference],
    }));
    assert_eq!(
        nested_inline.logical_digest().unwrap(),
        nested_reference.logical_digest().unwrap()
    );
}

#[test]
fn frozen_value_payload_v1_reference_is_rejected_before_use() {
    let reference = ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
        id: ValueId::derive(&[b"stale-value"]),
        logical_digest: Digest::sha256(b"stale-value"),
        encoded_length: 1,
        media_type: "application/runmat-value".into(),
        value_schema: "runmat-value/v1".into(),
        encryption_context: Digest::sha256(b"stale-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: "scope".into(),
        resident_fence: None,
    };
    let error = ValuePayload::Object(Box::new(reference))
        .validate(ValueLimits::default())
        .expect_err("value payload v1 must require re-encoding");
    assert!(matches!(
        error,
        runmat_execution::ContractError::UnsupportedSchema {
            actual: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
            supported: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        }
    ));
}

#[test]
fn pre_utf16_value_payload_schema_is_rejected_before_use() {
    let reference = ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V3,
        id: ValueId::derive(&[b"pre-utf16-value"]),
        logical_digest: Digest::sha256(b"pre-utf16-value"),
        encoded_length: 1,
        media_type: "application/runmat-value".into(),
        value_schema: "runmat-value/v3".into(),
        encryption_context: Digest::sha256(b"pre-utf16-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: "scope".into(),
        resident_fence: None,
    };
    let error = ValuePayload::Object(Box::new(reference))
        .validate(ValueLimits::default())
        .expect_err("pre-UTF-16 payloads must require re-encoding");
    assert!(matches!(
        error,
        runmat_execution::ContractError::UnsupportedSchema {
            actual: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V3,
            supported: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        }
    ));
}
