use super::*;
use runmat_types::*;

#[test]
fn plot_contract_preserves_unknown_series_and_checks_output_arity() {
    for (class, invalid) in [(standard::AXES, false), (standard::FIGURE, true)] {
        let target = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.into()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }));
        let call = CallRequest {
            arguments: vec![target, ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            !inference::infer(&call, &PLOT_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 1);
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let inferred = inference::infer(&request, &PLOT_CATALOG_ENTRY);
        assert_eq!(!inferred.diagnostics.is_empty(), count > 1);
        if count == 1 {
            assert_eq!(
                inferred.outputs[0],
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }))
            );
        }
        assert!(inferred.effects.0.contains(&EffectKind::UserInterface));
        assert!(inferred.effects.0.contains(&EffectKind::MaySuspend));
    }
    let empty = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    };
    assert!(!inference::infer(&empty, &PLOT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    assert_eq!(PLOT_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
}
