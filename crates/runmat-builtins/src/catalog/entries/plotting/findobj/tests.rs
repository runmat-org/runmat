use super::*;
use runmat_types::*;

#[test]
fn findobj_output_class_is_known_but_count_is_dynamic() {
    let result = inference::infer(
        &CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        },
        &FINDOBJ_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::Double)
    );
    assert!(matches!(result.outputs[0].shape, ShapeFact::Unknown));
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
}

#[test]
fn findobj_known_resident_control_does_not_prove_output() {
    let mut control = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    control.residency = ResidencyFact::Device { provider: None };
    let result = inference::infer(
        &CallRequest {
            arguments: vec![control],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        },
        &FINDOBJ_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
