use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, count: usize) -> CallInference {
    crate::infer_catalog_call(
        &STEP_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    )
}

fn model() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn default_response_is_double_host_column_and_data_form_has_no_ui_effect() {
    let inference = call(vec![model()], 2);
    assert!(
        inference.diagnostics.is_empty(),
        "{:?}",
        inference.diagnostics
    );
    assert_eq!(inference.outputs.len(), 2);
    for output in &inference.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(101), Some(1)]));
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    assert!(!inference.effects.0.contains(&EffectKind::UserInterface));
}

#[test]
fn known_other_class_and_excess_outputs_are_diagnosed() {
    let mut other = model();
    let ValueKindFact::Object(object) = &mut other.kind else {
        unreachable!()
    };
    object.runtime_class = None;
    object.class = Some(ClassId(7));
    assert!(!call(vec![other], 1).diagnostics.is_empty());
    assert!(!call(vec![model()], 3).diagnostics.is_empty());
}

#[test]
fn unresolved_plot_role_does_not_falsely_reject_later_model() {
    let result = call(
        vec![
            model(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            model(),
        ],
        0,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
}
