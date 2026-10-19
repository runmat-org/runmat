use super::*;
use runmat_types::*;

#[test]
fn colormap_inference_keeps_rgb_rows_and_trailing_singleton_runtime_forms() {
    for (shape, invalid, rows) in [
        (vec![Some(2), Some(3)], false, Some(2)),
        (vec![Some(3)], false, Some(1)),
        (vec![Some(2), Some(3), Some(1)], false, Some(2)),
        (vec![Some(2), Some(3), Some(2)], true, None),
        (vec![Some(0), Some(3)], true, None),
        (vec![None, Some(3)], false, None),
    ] {
        let map = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt8,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        );
        let result = inference::infer(&request(vec![map], vec![], 1), &COLORMAP_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), invalid);
        if !invalid {
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![rows, Some(3)])
            );
        }
    }
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn colormap_output_count_and_effects_are_independent_of_discarded_results() {
    for count in [0, 1, 2] {
        let result = inference::infer(&request(vec![], vec![], count), &COLORMAP_CATALOG_ENTRY);
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
fn colormap_unknown_input_stays_conservative_and_dense_text_is_rejected() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![unknown], vec![], 1), &COLORMAP_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let text_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(!inference::infer(
        &request(vec![text_array], vec![], 1),
        &COLORMAP_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
