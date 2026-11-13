use super::*;
use runmat_types::*;

#[test]
fn view_inference_rejects_nonfinite_camera_literals() {
    for value in [f64::NAN, f64::INFINITY, f64::MAX] {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(value),
                LiteralValue::Number(0.),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        assert!(!inference::infer(&request, &VIEW_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}

#[test]
fn view_inference_tracks_output_dependent_shape_and_effects() {
    for count in [0, 1, 2, 3] {
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &VIEW_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 2);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        if count == 1 {
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(1), Some(2)])
            );
        }
        if count == 2 {
            assert!(result
                .outputs
                .iter()
                .all(|output| output.shape == ShapeFact::Scalar
                    && output.residency == ResidencyFact::Host));
        }
    }
}

#[test]
fn view_inference_preserves_runtime_two_element_nd_form_and_unknowns() {
    for (shape, invalid) in [
        (ShapeFact::Unknown, false),
        (ShapeFact::from(vec![Some(1), Some(1), Some(2)]), false),
        (ShapeFact::from(vec![Some(3), Some(1)]), true),
        (ShapeFact::from(vec![Some(0), Some(0)]), true),
    ] {
        let mut arg = ValueFact::unknown(DynamicReason::RuntimeValue);
        arg.shape = shape;
        let request = CallRequest {
            arguments: vec![arg],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        assert_eq!(
            !inference::infer(&request, &VIEW_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
}
