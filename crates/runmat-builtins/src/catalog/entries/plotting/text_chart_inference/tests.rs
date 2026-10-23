use super::*;

#[test]
fn chart_handles_output_selection_and_effects_are_declared() {
    for (entry, minimum) in [
        (&super::super::textscatter::TEXTSCATTER_CATALOG_ENTRY, 2),
        (&super::super::textscatter3::TEXTSCATTER3_CATALOG_ENTRY, 2),
        (&super::super::wordcloud::WORDCLOUD_CATALOG_ENTRY, 1),
    ] {
        let mut request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); minimum],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, entry, minimum);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
        assert!(entry.contract.effects.contains(&EffectKind::UserInterface));
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer(&request, entry, minimum).diagnostics.is_empty());
        request.arguments.clear();
        assert!(infer(&request, entry, minimum)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-TEXT-CHART-ARITY"));
    }
}
