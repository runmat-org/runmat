use super::*;
use futures::executor::block_on;
fn call(args: Vec<Value>) -> BuiltinResult<crate::execution::RootedValue> {
    block_on(integer_constraint_builtin_registered(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}
#[test]
fn constructor_schema_is_disjoint_and_cow_edits_cannot_reuse_stale_validation() {
    let original = call(vec![
        Value::from("Integer"),
        Value::Num(1.0),
        Value::from("SemiContinuous"),
        Value::Num(2.0),
    ])
    .unwrap();
    let Value::Object(object) = original.value() else {
        panic!("domain object")
    };
    assert!(schema::validate(object, None).is_ok());
    let admitted = schema::ValidatedConstraint::admit(
        object,
        &crate::execution::value_codec::ValidationWorkspaceOwner::for_runtime(None),
    )
    .unwrap();
    let mut changed = object.clone();
    changed.replace_property_unbounded("SemiInteger".into(), Value::Num(1.0));
    assert!(schema::validate(&changed, None).is_err());
    assert_eq!(
        admitted
            .indices(schema::Property::SemiInteger)
            .unwrap()
            .len(),
        0
    );
    assert!(schema::validate(object, None).is_ok());
    changed.replace_property_unbounded("SemiInteger".into(), Value::from("not an index"));
    assert!(schema::validate(&changed, None).is_err());
    assert!(call(vec![
        Value::from("Integer"),
        Value::Num(1.0),
        Value::from("SemiInteger"),
        Value::Num(1.0)
    ])
    .is_err());
    assert!(call(vec![
        Value::from("Integer"),
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap())
    ])
    .is_err());
    assert!(call(vec![Value::from("Integer"), Value::Num(1.5)]).is_err());
    assert!(call(vec![Value::from("Other"), Value::Num(1.0)]).is_err());
}
#[test]
fn registered_inline_schema_roundtrips_and_rejects_current_invalid_generation() {
    let original = call(vec![
        Value::from("SemiInteger"),
        Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![1, 2]).unwrap()),
    ])
    .unwrap();
    let payload = crate::execution::value_codec::encode_inline_value(original.value()).unwrap();
    let restored = crate::execution::value_codec::decode_inline_value(&payload).unwrap();
    assert_eq!(&restored, original.value());
    let Value::Object(mut object) = restored else {
        panic!("domain object")
    };
    object.replace_property_unbounded("Integer".into(), Value::Num(3.0));
    assert!(crate::execution::value_codec::encode_inline_value(&Value::Object(object)).is_err());
}

#[test]
fn graph_roundtrip_and_mat_publication_use_the_current_schema_not_property_claims() {
    use crate::execution::value_codec::{
        decode_payload_graph_in_context, encode_payload_graph_in_context,
    };
    use runmat_execution::{Digest, ExecutableSourceMapV2, ExecutableSourceRecordV2};
    let original = call(vec![Value::from("SemiContinuous"), Value::Num(1.0)]).unwrap();
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let issuer = crate::execution::value_codec::GraphSourceIssuer::capture_session(&runtime);
    assert!(matches!(
        crate::execution::value_codec::prepare_payload_graph(
            &[original.value().clone()],
            &issuer,
            runmat_execution::value::ValueLimits::default(),
            None
        ),
        Err(crate::execution::value_codec::ValueCodecError::Allocation {
            source: crate::context::host_allocation::HostAllocationError::MissingManagedOwner,
            ..
        })
    ));
    let workspace = crate::execution::value_codec::ValidationWorkspaceOwner::capture(&runtime);
    assert!(crate::execution::value_codec::prepare_payload_graph(
        &[original.value().clone()],
        &issuer,
        runmat_execution::value::ValueLimits::default(),
        Some(&workspace)
    )
    .is_ok());
    let sources = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
        source_id: runmat_types::ProgramSourceId(0),
        owner_identity: "domain-codec-test".into(),
        relative_path: "domain.m".into(),
        content_revision: Digest::sha256(b"domain"),
        display_name: "domain.m".into(),
        text: "domain".into(),
    }])
    .unwrap();
    let graph = encode_payload_graph_in_context(
        &[original.value().clone()],
        &runtime,
        &sources,
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        runmat_execution::value::ValueLimits::default(),
    )
    .unwrap();
    let restored = decode_payload_graph_in_context(
        &graph,
        &runtime,
        &sources,
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        runmat_execution::value::ValueLimits::default(),
    )
    .unwrap();
    assert_eq!(restored.values(), &[original.value().clone()]);
    let mut tampered = graph;
    let node = tampered
        .nodes
        .iter_mut()
        .find_map(|node| match node {
            runmat_execution::value::GraphNode::Registered(class) => Some(class),
            _ => None,
        })
        .unwrap();
    node.schema_version += 1;
    assert!(decode_payload_graph_in_context(
        &tampered,
        &runtime,
        &sources,
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        runmat_execution::value::ValueLimits::default()
    )
    .is_err());

    use crate::builtins::introspection::object_serialization::{
        reconcile_mat_object_product, resolve_mat_object_class,
    };
    let (class, product) = block_on(resolve_mat_object_class(
        schema::CLASS.display_name(),
        Some(&runtime),
    ))
    .unwrap();
    let Value::Object(mut invalid) = original.value().clone() else {
        panic!("object")
    };
    invalid.replace_property_unbounded("Integer".into(), Value::Num(1.0));
    let invalid =
        crate::execution::RootedValue::in_optional_context(Value::Object(invalid), Some(&runtime))
            .unwrap();
    assert!(
        reconcile_mat_object_product(invalid, &class, &product, false, Some(&runtime)).is_err()
    );
    let valid = crate::execution::RootedValue::in_optional_context(
        original.value().clone(),
        Some(&runtime),
    )
    .unwrap();
    assert!(reconcile_mat_object_product(valid, &class, &product, false, Some(&runtime)).is_ok());
}
