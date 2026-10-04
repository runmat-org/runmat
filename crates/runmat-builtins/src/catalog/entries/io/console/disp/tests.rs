use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

#[test]
fn one_any_input_and_no_public_output() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Unknown)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    let inferred = inference::infer(&request, &DISP_CATALOG_ENTRY);
    assert!(inferred.outputs.is_empty());
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
}
