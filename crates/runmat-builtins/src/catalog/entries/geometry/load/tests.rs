use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact,
    ValueKindFact,
};

fn request(input: ValueFact, count: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(count),
    }
}

#[test]
fn load_preserves_typed_asset_and_rejects_numeric_path() {
    let valid = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            RequestedOutputCount::One,
        ),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let ValueKindFact::Object(object) = &valid.outputs[0].kind else {
        panic!("expected asset object")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&GEOMETRY_ASSET_CLASS.owned())
    );
    let unknown = inference::infer(
        &request(
            ValueFact::unknown(DynamicReason::RuntimeValue),
            RequestedOutputCount::One,
        ),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
    let invalid = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::Logical),
            RequestedOutputCount::Exactly(2),
        ),
        &GEOMETRY_LOAD_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GEOMETRY-LOAD-PATH"));
    assert!(!invalid.diagnostics.is_empty());
}
