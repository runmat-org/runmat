use super::*;
use runmat_types::*;

#[test]
fn xmlwrite_output_and_filesystem_capability_depend_on_form() {
    let request = |arguments| CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let text = ValueFact::scalar(ValueKindFact::String);
    let pure = inference::infer(&request(vec![text.clone()]), &XMLWRITE_CATALOG_ENTRY);
    assert_eq!(pure.outputs[0].kind, ValueKindFact::Character);
    assert!(!pure
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    let file = inference::infer(&request(vec![text.clone(), text]), &XMLWRITE_CATALOG_ENTRY);
    assert!(matches!(file.outputs[0].kind, ValueKindFact::Numeric(_)));
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}
