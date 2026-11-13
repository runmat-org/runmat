use super::*;
use runmat_types::*;
#[test]
fn daspect_inference_distinguishes_mode_query_and_statement_setter() {
    for (mode, count, invalid) in [
        ("mode", 1, false),
        ("auto", 0, false),
        ("manual", 1, true),
        ("bad", 0, true),
        ("mode", 2, true),
    ] {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String(mode.into())]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &DASPECT_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), invalid);
        if mode == "mode" && count == 1 {
            assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        }
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
    }
}
#[test]
fn daspect_inference_preserves_unknowns_and_strict_vector_rank() {
    for (shape, invalid) in [
        (ShapeFact::Unknown, false),
        (ShapeFact::from(vec![Some(1), Some(3)]), false),
        (ShapeFact::from(vec![Some(1), Some(1), Some(3)]), true),
        (ShapeFact::from(vec![Some(0), Some(0)]), true),
    ] {
        let arg = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![arg],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        assert_eq!(
            !inference::infer(&request, &DASPECT_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
}
