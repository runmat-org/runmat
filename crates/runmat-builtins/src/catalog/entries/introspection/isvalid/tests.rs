use super::*;
use runmat_types::*;

#[test]
fn numeric_and_resident_input_reject_before_claiming_a_logical_result() {
    let mut integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    for residency in [
        ResidencyFact::Host,
        ResidencyFact::Device { provider: None },
    ] {
        integer.residency = residency;
        let inferred = inference::infer(
            &CallRequest {
                arguments: vec![integer.clone()],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &ISVALID_CATALOG_ENTRY,
        );
        assert!(!inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    }
}
