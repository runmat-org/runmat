use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    )
}
fn call(input: ValueFact, control: ValueFact, outputs: RequestedOutputCount) -> CallInference {
    crate::infer_catalog_call(
        &SPECTROGRAM_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![input, control],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
            outputs: OutputSelection::new(outputs),
        },
    )
}
#[test]
fn every_named_output_uses_signal_class_not_control_class() {
    for class in [NumericClass::Single, NumericClass::Double] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            for control_class in [
                NumericClass::Single,
                NumericClass::Double,
                NumericClass::Int16,
            ] {
                let result = call(
                    numeric(class, domain),
                    numeric(control_class, NumericDomain::Real),
                    RequestedOutputCount::Exactly(4),
                );
                assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
                for (slot, output) in result.outputs.iter().enumerate() {
                    assert_eq!(output.numeric().unwrap().class, class);
                    assert_eq!(
                        output.numeric().unwrap().domain,
                        if slot == 0 {
                            NumericDomain::Complex
                        } else {
                            NumericDomain::Real
                        }
                    );
                    assert_eq!(output.residency, ResidencyFact::Host);
                }
            }
        }
    }
}
#[test]
fn unknown_signal_stays_unknown_and_resident_capabilities_are_not_invented() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = call(
        unknown.clone(),
        numeric(NumericClass::Single, NumericDomain::Real),
        RequestedOutputCount::Exactly(4),
    );
    assert!(result
        .outputs
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));
    let mut signal = numeric(NumericClass::Single, NumericDomain::Complex);
    signal.residency = ResidencyFact::Device {
        provider: Some("actual-owner".into()),
    };
    let result = call(signal, unknown, RequestedOutputCount::Exactly(4));
    for output in result.outputs {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Single);
        assert_eq!(output.residency, ResidencyFact::Unknown);
        assert_eq!(output.layout, LayoutFact::Unknown);
        assert_eq!(output.contiguity, ContiguityFact::Unknown);
    }
}
#[test]
fn surplus_padding_is_host_double_while_zero_and_dynamic_counts_stay_correct() {
    let signal = numeric(NumericClass::Single, NumericDomain::Real);
    let control = numeric(NumericClass::Double, NumericDomain::Real);
    let result = call(
        signal.clone(),
        control.clone(),
        RequestedOutputCount::Exactly(6),
    );
    for output in &result.outputs[..4] {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Single);
    }
    for output in &result.outputs[4..] {
        assert_eq!(
            output.numeric().unwrap(),
            NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            }
        );
        assert!(output.is_scalar());
        assert_eq!(output.residency, ResidencyFact::Host);
        assert_eq!(output.storage, StorageFact::Scalar);
    }
    assert!(
        call(signal.clone(), control.clone(), RequestedOutputCount::Zero)
            .outputs
            .is_empty()
    );
    assert!(
        call(
            signal,
            control,
            RequestedOutputCount::CurrentFunctionNargout
        )
        .dynamic_outputs
    );
}
#[test]
fn admitted_integer_and_logical_signals_have_double_outputs() {
    for signal in [
        numeric(NumericClass::Int64, NumericDomain::Real),
        ValueFact::scalar(ValueKindFact::Logical),
    ] {
        let result = call(
            signal,
            numeric(NumericClass::Single, NumericDomain::Real),
            RequestedOutputCount::Exactly(4),
        );
        assert!(result
            .outputs
            .iter()
            .all(|output| output.numeric().unwrap().class == NumericClass::Double));
    }
    assert!(!call(
        numeric(NumericClass::Int16, NumericDomain::Complex),
        ValueFact::unknown(DynamicReason::RuntimeValue),
        RequestedOutputCount::One
    )
    .diagnostics
    .is_empty());
    assert_eq!(
        SPECTROGRAM_INTEGER_CAPABILITIES[1].output_class,
        crate::BuiltinIntegerOutputClassRule::FunctionSpecific
    );
}
#[test]
fn canonical_entry_owns_descriptor_documentation_and_inference() {
    assert!(crate::validate_builtin_catalog(&[&SPECTROGRAM_CATALOG_ENTRY]).is_empty());
    let entry =
        crate::builtin_catalog_entry_by_name("spectrogram").expect("composed canonical identity");
    assert_eq!(entry.identity.name, "spectrogram");
    assert_eq!(entry.descriptor.signatures[1].outputs.len(), 4);
}
