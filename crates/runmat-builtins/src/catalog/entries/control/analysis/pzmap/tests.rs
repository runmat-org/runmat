use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn model() -> ValueFact {
    let mut properties = BTreeMap::new();
    for name in ["Numerator", "Denominator"] {
        properties.insert(
            name.into(),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(1)]),
                StorageFact::Dense,
            ),
        );
    }
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
fn call(args: Vec<ValueFact>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &PZMAP_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn data_forms_have_host_columns_without_ui_effects() {
    let result = call(vec![model()], 2);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 2);
    for output in result.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(0), Some(1)]));
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    assert!(!result.effects.0.contains(&EffectKind::UserInterface));
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
}

#[test]
fn statement_form_has_ui_effects_and_no_data_outputs() {
    let result = call(
        vec![model(), ValueFact::scalar(ValueKindFact::String), model()],
        0,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.outputs.is_empty());
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
}

#[test]
fn invalid_class_arity_and_output_count_are_diagnosed() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!call(vec![numeric], 1).diagnostics.is_empty());
    assert!(!call(vec![model(), model()], 1).diagnostics.is_empty());
    assert!(!call(vec![model()], 3).diagnostics.is_empty());
    assert!(call(
        vec![
            model(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String)
        ],
        0
    )
    .diagnostics
    .iter()
    .any(|diagnostic| diagnostic.code == "RM-CATALOG-PZMAP-PLOT-STYLE"));
}

#[test]
fn known_other_class_and_dynamic_output_count_are_handled_soundly() {
    let mut other = model();
    let ValueKindFact::Object(object) = &mut other.kind else {
        unreachable!()
    };
    object.class = Some(ClassId(8));
    object.runtime_class = None;
    assert!(!call(vec![other], 1).diagnostics.is_empty());

    let result = crate::infer_catalog_call(
        &PZMAP_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![model(), model()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::CurrentFunctionNargout),
        },
    );
    assert!(!result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PZMAP-ARITY"));
}
