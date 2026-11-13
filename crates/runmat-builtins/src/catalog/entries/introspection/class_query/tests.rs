use super::*;
use runmat_types::*;
use std::collections::BTreeSet;

#[test]
fn class_is_host_string_scalar_for_known_unknown_and_resident_input() {
    for input in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        })),
        ValueFact::unknown(DynamicReason::RuntimeValue),
        {
            let mut resident = ValueFact::scalar(ValueKindFact::Logical);
            resident.residency = ResidencyFact::Device { provider: None };
            resident
        },
    ] {
        let inferred = inference::infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &CLASS_CATALOG_ENTRY,
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(
            inferred.outputs[0],
            ValueFact::scalar(ValueKindFact::String)
        );
    }
}

#[test]
fn class_wrong_arity_does_not_infer_a_valid_result() {
    let inferred = inference::infer(
        &CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &CLASS_CATALOG_ENTRY,
    );
    assert!(!inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn class_output_selection_preserves_carrier_and_validates_discards() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let infer = |outputs| {
        inference::infer(
            &CallRequest {
                arguments: vec![input.clone()],
                literals: LiteralContext::default(),
                outputs,
            },
            &CLASS_CATALOG_ENTRY,
        )
    };
    let discarded = infer(OutputSelection::new(RequestedOutputCount::Zero));
    assert!(discarded.outputs.is_empty());
    assert!(discarded.diagnostics.is_empty());
    let selected = infer(OutputSelection {
        requested: RequestedOutputCount::One,
        discarded: BTreeSet::from([0]),
    });
    assert_eq!(
        selected.outputs,
        vec![ValueFact::scalar(ValueKindFact::String)]
    );
    assert_eq!(selected.discarded, vec![0]);
    assert!(selected.diagnostics.is_empty());
    let invalid_discard = infer(OutputSelection {
        requested: RequestedOutputCount::One,
        discarded: BTreeSet::from([1]),
    });
    assert!(invalid_discard
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-DISCARD"));
    let too_many = infer(OutputSelection::new(RequestedOutputCount::Exactly(2)));
    assert!(too_many
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
