use super::*;
use runmat_types::*;

#[test]
fn source_modes_have_string_carrier_and_fixed_output_count() {
    for arguments in [
        vec![],
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }))],
    ] {
        let request = CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result = inference::infer(&request, &MFILENAME_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs,
            vec![ValueFact::scalar(ValueKindFact::String)]
        );
    }
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    assert!(inference::infer(&request, &MFILENAME_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn too_many_options_is_invalid_but_zero_outputs_is_admitted() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &MFILENAME_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MFILENAME-ARITY"));
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    assert!(inference::infer(&request, &MFILENAME_CATALOG_ENTRY)
        .outputs
        .is_empty());
}
