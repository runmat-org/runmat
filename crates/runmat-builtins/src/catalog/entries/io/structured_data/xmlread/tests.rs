use super::*;
use runmat_types::*;

#[test]
fn xmlread_infers_document_class_and_rejects_numeric_filename() {
    let request = |kind| CallRequest {
        arguments: vec![ValueFact::scalar(kind)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request(ValueKindFact::String), &XMLREAD_CATALOG_ENTRY);
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("document object")
    };
    assert_eq!(
        object.runtime_class,
        Some(super::super::XML_DOCUMENT_CLASS.owned())
    );
    assert!(object.properties.contains_key("DocumentElement"));
    assert!(
        !inference::infer(&request(ValueKindFact::Logical), &XMLREAD_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
