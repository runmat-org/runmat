use super::*;
use runmat_types::*;
#[test]
fn table_stats_and_literal_off_effects_are_authored_without_numerical_value_guesses() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 3],
        literals: LiteralContext {
            literal_args: vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Character("off".into()),
            ],
        },
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
    };
    let result = inference::infer(&request, &ANOVA1_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(4), Some(6)])
    );
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Cell(_)));
    let ValueKindFact::Struct(stats) = &result.outputs[2].kind else {
        panic!("stats")
    };
    assert!(stats.fields_complete);
    assert_eq!(stats.fields.len(), 6);
    for name in ["gnames", "n", "source", "means", "df", "s"] {
        assert!(stats.fields.contains_key(name));
    }
    for effect in [
        EffectKind::FilesystemWrite,
        EffectKind::UserInterface,
        EffectKind::HostCallback,
    ] {
        assert!(!result.effects.0.contains(&effect));
    }
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    request.literals = LiteralContext::default();
    let result = inference::infer(&request, &ANOVA1_CATALOG_ENTRY);
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
    assert_eq!(ANOVA1_CATALOG_ENTRY.documentation.examples.len(), 2);
    assert!(ANOVA1_CATALOG_ENTRY.integer_capabilities.is_empty());
}
