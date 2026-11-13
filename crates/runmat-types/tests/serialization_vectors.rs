use runmat_types::codec::{decode_canonical, encode_canonical, CanonicalCodecError};
use runmat_types::{
    infer_binary, infer_call, infer_index, CallContract, CallRequest, CallableFact, ClassKind,
    DynamicReason, ExternalClassDeclaration, IndexKind, IndexResultContext, IndexSelectorFact,
    NumericClass, NumericDomain, NumericFact, OperatorKind, OutputSelection, QualifiedName,
    RequestedOutputCount, ShapeFact, SymbolName, ValueFact, ValueKindFact,
};

#[test]
fn canonical_encoding_is_deterministic_and_round_trips() {
    let fact = ValueFact::unknown(DynamicReason::ForeignBoundary);
    let first = encode_canonical(&fact).unwrap();
    let second = encode_canonical(&fact).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        std::str::from_utf8(&first).unwrap(),
        r#"{"schema":"runmat-types","major":1,"minor":3,"fact":{"kind":"Unknown","shape":"Unknown","storage":"Unknown","layout":"Unknown","contiguity":"Unknown","view":"Unknown","residency":"Unknown","alias":"Unknown","mutation":"Unknown","certainty":{"Dynamic":"ForeignBoundary"},"invalidation":[]}}"#
    );
    assert_eq!(decode_canonical(&first).unwrap(), fact);
}

#[test]
fn canonical_schema_minor_compatibility_is_backward_only() {
    let fact = ValueFact::unknown(DynamicReason::ForeignBoundary);
    let encoded = encode_canonical(&fact).unwrap();
    let mut envelope: serde_json::Value = serde_json::from_slice(&encoded).unwrap();

    for minor in 0..=3 {
        envelope["minor"] = serde_json::json!(minor);
        let compatible = serde_json::to_vec(&envelope).unwrap();
        assert_eq!(decode_canonical(&compatible).unwrap(), fact, "v1.{minor}");
    }

    envelope["minor"] = serde_json::json!(4);
    let future = serde_json::to_vec(&envelope).unwrap();
    assert!(matches!(
        decode_canonical(&future),
        Err(CanonicalCodecError::UnsupportedSchema {
            schema,
            major: 1,
            minor: 4,
        }) if schema == "runmat-types"
    ));
}

#[test]
fn callable_facts_from_schema_1_0_decode_without_inventing_capabilities() {
    let encoded = br#"{
        "identity": null,
        "parameters": [],
        "parameters_complete": true,
        "outputs": [],
        "outputs_complete": true,
        "variadic_inputs": false,
        "variadic_outputs": false,
        "captures": [],
        "captures_complete": true
    }"#;
    let callable: CallableFact = serde_json::from_slice(encoded).unwrap();
    assert!(callable.capabilities.0.is_empty());
}

#[test]
fn immutable_declarations_round_trip_without_runtime_state() {
    let declaration = ExternalClassDeclaration {
        name: QualifiedName(vec![
            SymbolName("example".into()),
            SymbolName("Widget".into()),
        ]),
        parent: Some(QualifiedName(vec![SymbolName("handle".into())])),
        kind: ClassKind::Handle,
        is_sealed: true,
        is_abstract: false,
        properties: Vec::new(),
        methods: Vec::new(),
        array_shape: None,
    };
    let encoded = serde_json::to_vec(&declaration).unwrap();
    let decoded: ExternalClassDeclaration = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, declaration);
}

#[test]
fn legacy_class_declarations_default_to_scalar_object_geometry() {
    let legacy = serde_json::json!({
        "name": ["example", "Widget"], "parent": null, "kind": "Value",
        "is_sealed": false, "is_abstract": false, "properties": [], "methods": []
    });
    let mut declaration: ExternalClassDeclaration = serde_json::from_value(legacy).unwrap();
    assert_eq!(declaration.array_shape, None);
    assert!(serde_json::to_value(&declaration)
        .unwrap()
        .get("array_shape")
        .is_none());
    declaration.array_shape = Some(runmat_types::ClassArrayShape::ConsistentProperties {
        primary: "data".into(),
        others: vec!["mask".into()],
        payload: runmat_types::ClassArrayPayload::NumericOrLogical,
    });
    let encoded = serde_json::to_vec(&declaration).unwrap();
    let decoded: ExternalClassDeclaration = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, declaration);
}

fn semantic_rule_parity_vectors() {
    let numeric = |shape| {
        let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        fact.shape = shape;
        fact
    };
    let broadcast = infer_binary(
        OperatorKind::ElementwiseMultiply,
        &numeric(ShapeFact::from(vec![Some(2), Some(1)])),
        &numeric(ShapeFact::from(vec![Some(1), Some(3)])),
    );
    assert_eq!(
        broadcast.fact.shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );

    let indexed = infer_index(
        &numeric(ShapeFact::from(vec![Some(2), Some(3)])),
        IndexKind::Paren,
        &[IndexSelectorFact::Colon],
        IndexResultContext::ReadSingle,
    );
    assert_eq!(indexed.fact.shape, ShapeFact::from(vec![Some(6), Some(1)]));

    let call = infer_call(
        &CallContract::fixed(vec![numeric(ShapeFact::Scalar)]),
        &CallRequest {
            arguments: Vec::new(),
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(call.outputs.len(), 1);
    assert!(!call.dynamic_outputs);
}

#[test]
fn semantic_rules_have_stable_native_vectors() {
    semantic_rule_parity_vectors();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_uses_the_same_canonical_vector() {
    canonical_encoding_is_deterministic_and_round_trips();
    immutable_declarations_round_trip_without_runtime_state();
    semantic_rule_parity_vectors();
}
