use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

#[test]
fn series_declares_typed_model_and_output_boundary() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue); 2],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &SERIES_CATALOG_ENTRY);
    assert_eq!(result.outputs.len(), 1);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(SERIES_DESCRIPTOR.signatures.len(), 3);
}
