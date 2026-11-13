use super::*;
use runmat_types::*;
#[test]
fn axes_inference_returns_host_handle_without_inventing_graphics_identity() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].identity.name, "axes");
    for count in [0, 1, 2] {
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        };
        let result = inference::infer(&request, &AXES_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        if count == 1 {
            assert_eq!(
                result.outputs[0],
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real
                }))
            );
        }
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
    }
}
#[test]
fn axes_inference_validates_known_property_syntax_and_preserves_unknown_target() {
    for (class, invalid) in [(standard::AXES, false), (standard::FIGURE, false)] {
        let target = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.into()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }));
        let call = CallRequest {
            arguments: vec![target],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            !inference::infer(&call, &AXES_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
    let text = ValueFact::scalar(ValueKindFact::String);
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (arguments, invalid) in [
        (vec![text.clone()], true),
        (vec![text, unknown.clone()], false),
        (vec![unknown], false),
    ] {
        let request = CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            !inference::infer(&request, &AXES_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            invalid
        );
    }
}
