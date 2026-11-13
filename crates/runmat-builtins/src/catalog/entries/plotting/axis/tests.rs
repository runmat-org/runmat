use super::*;
use runmat_types::*;

#[test]
fn axis_mode_character_shape_rejection_is_conservative() {
    for (shape, rejected) in [
        (vec![Some(2), Some(2)], true),
        (vec![Some(1), Some(2), Some(2)], true),
        (vec![Some(1), Some(4), Some(1)], false),
        (vec![None, Some(4), None], false),
    ] {
        let fact = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(shape),
            StorageFact::Dense,
        );
        let result = inference::infer(&request(vec![fact], vec![], 0), &AXIS_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), rejected);
    }
}

#[test]
fn axis_query_is_dynamic_length_while_known_setters_return_logical() {
    let result = inference::infer(&request(vec![], vec![], 1), &AXIS_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let limits = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let result = inference::infer(&request(vec![limits], vec![], 1), &AXIS_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let result = inference::infer(&request(vec![scalar], vec![], 1), &AXIS_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn axis_output_count_and_effects_are_independent_of_discarded_results() {
    for count in [0, 1, 2] {
        let result = inference::infer(&request(vec![], vec![], count), &AXIS_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        assert!(result.effects.0.contains(&EffectKind::HostCallback));
        assert!(result.effects.0.contains(&EffectKind::MayThrow));
        if count == 0 {
            assert!(result.outputs.is_empty());
        }
    }
}

#[test]
fn axis_unknown_input_stays_conservative_and_dense_text_is_rejected() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![unknown], vec![], 1), &AXIS_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let text_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(
        !inference::infer(&request(vec![text_array], vec![], 1), &AXIS_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
