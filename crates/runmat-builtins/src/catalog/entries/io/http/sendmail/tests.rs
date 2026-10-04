use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>) -> CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &SENDMAIL_CATALOG_ENTRY,
    )
}

#[test]
fn successful_mail_status_is_proven_double_and_effectful() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = call(vec![text.clone(), text]);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    assert!(inferred.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(inferred.effects.0.contains(&EffectKind::EnvironmentRead));
}

#[test]
fn known_invalid_forms_reject_and_unknown_remains_possible() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(!call(vec![integer, text.clone()]).diagnostics.is_empty());
    assert!(call(vec![
        ValueFact::unknown(DynamicReason::RuntimeValue),
        text.clone()
    ])
    .diagnostics
    .is_empty());
    assert!(call(vec![text.clone(), text.clone(), text.clone(), text])
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SENDMAIL-ATTACHMENTS"));
}
