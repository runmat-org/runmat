use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn stopwatch_facts_and_token_rejection() {
    let started = inference::tic(&request(Vec::new()), &TIC_CATALOG_ENTRY);
    assert!(matches!(
        &started.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            ..
        })
    ));
    let token = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let elapsed = inference::toc(&request(vec![token]), &TOC_CATALOG_ENTRY);
    assert!(matches!(
        &elapsed.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    let invalid = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(inference::toc(&request(vec![invalid]), &TOC_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TOC-TOKEN"));
}
