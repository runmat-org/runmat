use super::*;
use runmat_types::*;
#[test]
fn universal_type_query_returns_host_logical_for_known_unknown_and_device_facts() {
    for mut input in [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ] {
        input.residency = ResidencyFact::Device { provider: None };
        let inferred = inference::infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &ISOBJECT_CATALOG_ENTRY,
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(
            inferred.outputs[0],
            ValueFact::scalar(ValueKindFact::Logical)
        );
    }
}
