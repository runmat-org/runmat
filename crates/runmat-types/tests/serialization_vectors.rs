use runmat_types::codec::{decode_canonical, encode_canonical};
use runmat_types::{
    ClassKind, DynamicReason, ExternalClassDeclaration, QualifiedName, SymbolName, ValueFact,
};

#[test]
fn canonical_encoding_is_deterministic_and_round_trips() {
    let fact = ValueFact::unknown(DynamicReason::ForeignBoundary);
    let first = encode_canonical(&fact).unwrap();
    let second = encode_canonical(&fact).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        std::str::from_utf8(&first).unwrap(),
        r#"{"schema":"runmat-types","major":1,"minor":0,"fact":{"kind":"Unknown","shape":"Unknown","storage":"Unknown","layout":"Unknown","contiguity":"Unknown","view":"Unknown","residency":"Unknown","alias":"Unknown","mutation":"Unknown","certainty":{"Dynamic":"ForeignBoundary"},"invalidation":[]}}"#
    );
    assert_eq!(decode_canonical(&first).unwrap(), fact);
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
    };
    let encoded = serde_json::to_vec(&declaration).unwrap();
    let decoded: ExternalClassDeclaration = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, declaration);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_uses_the_same_canonical_vector() {
    canonical_encoding_is_deterministic_and_round_trips();
    immutable_declarations_round_trip_without_runtime_state();
}
