use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
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

    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Complex,
    }));
    let mut resident = complex.clone();
    resident.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    });
    resident.residency = ResidencyFact::Device { provider: None };
    let mut sparse = resident.clone();
    sparse.residency = ResidencyFact::Host;
    sparse.storage = StorageFact::Sparse;
    for invalid in [complex, resident, sparse] {
        assert!(inference::toc(&request(vec![invalid]), &TOC_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-TOC-TOKEN"));
    }
}

#[test]
fn toc_console_effect_is_only_required_for_unobserved_result() {
    let mut unobserved = request(Vec::new());
    unobserved.outputs = OutputSelection::new(RequestedOutputCount::Zero);
    let printed = inference::toc(&unobserved, &TOC_CATALOG_ENTRY);
    assert!(printed
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
    assert!(printed
        .effects
        .0
        .contains(&runmat_types::EffectKind::HostCallback));
    assert!(printed
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::UserInterface));

    let observed = inference::toc(&request(Vec::new()), &TOC_CATALOG_ENTRY);
    assert!(!observed
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
    assert!(!observed
        .effects
        .0
        .contains(&runmat_types::EffectKind::HostCallback));
    assert!(!observed
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::UserInterface));
}
