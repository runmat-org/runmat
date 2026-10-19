use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

#[test]
fn no_output_any_input_and_effectful_callback_boundary() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Unknown)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    let inferred = inference::infer(&request, &DISPLAY_CATALOG_ENTRY);
    assert!(inferred.outputs.is_empty());
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred
        .effects
        .0
        .contains(&runmat_types::EffectKind::HostCallback));
    assert!(inferred
        .effects
        .0
        .contains(&runmat_types::EffectKind::Unknown));
    assert!(inferred
        .effects
        .0
        .contains(&runmat_types::EffectKind::MaySuspend));
    let primitive = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(
            runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            },
        ))],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    let primitive = inference::infer(&primitive, &DISPLAY_CATALOG_ENTRY);
    assert!(!primitive
        .effects
        .0
        .contains(&runmat_types::EffectKind::Unknown));
    assert!(!primitive
        .effects
        .0
        .contains(&runmat_types::EffectKind::HostCallback));
    assert!(primitive
        .effects
        .0
        .contains(&runmat_types::EffectKind::FilesystemWrite));
}
