use runmat_execution::identity::{Digest, NodeLeaseId, ValueId, WorkerId};
use runmat_execution::value::{
    DenseValue, ElementType, InlineValue, RegisteredData, RegisteredField, ResidentFence,
    SparseValue, ValueLimits, ValuePayload, ValueRef, ValueRefKind,
};

#[test]
fn nested_payloads_obey_depth_and_node_limits() {
    let payload = ValuePayload::Inline(Box::new(InlineValue::Cell(vec![ValuePayload::Inline(
        Box::new(InlineValue::Cell(vec![ValuePayload::Inline(Box::new(
            InlineValue::Logical(true),
        ))])),
    )])));
    assert!(payload
        .validate(ValueLimits {
            max_depth: 1,
            ..ValueLimits::default()
        })
        .is_err());
    assert!(payload.validate(ValueLimits::default()).is_ok());
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
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
        id: ValueId::derive(&[b"value"]),
        logical_digest: Digest::sha256(b"value"),
        encoded_length: 1,
        media_type: "application/runmat-value".into(),
        value_schema: "runmat-value/v1".into(),
        encryption_context: Digest::sha256(b"context"),
        kind: ValueRefKind::ResidentObject,
        authorization_scope: "scope".into(),
        resident_fence: None,
    };
    assert!(ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .is_err());

    reference.resident_fence = Some(ResidentFence {
        worker_id: WorkerId::derive(&[b"worker"]),
        node_lease_id: NodeLeaseId::derive(&[b"lease"]),
        process_generation: 2,
        device_identity: Some("gpu-0".into()),
    });
    assert!(ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .is_ok());

    reference.kind = ValueRefKind::ResultObject;
    assert!(ValuePayload::Object(Box::new(reference))
        .validate(ValueLimits::default())
        .is_err());
}
