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
                ShapeFact::from(vec![Some(1), Some(2)]),
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
fn gains(count: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(count)]),
        StorageFact::Dense,
    )
}
fn call(args: Vec<ValueFact>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &RLOCUS_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn explicit_gain_count_shapes_outputs_without_inventing_branch_count() {
    let result = call(vec![model(), gains(3)], 2);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(!result.effects.0.contains(&EffectKind::UserInterface));
}

#[test]
fn extra_outputs_match_runtime_zero_padding() {
    let result = call(vec![model(), gains(1)], 3);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[2].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
}

#[test]
fn statement_form_has_ui_effects_and_rejects_non_model_first_input() {
    let result = call(vec![model(), gains(3)], 0);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.outputs.is_empty());
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
    assert!(!call(vec![gains(3)], 0).diagnostics.is_empty());
}

#[test]
fn empty_or_matrix_gain_is_diagnosed() {
    assert!(!call(vec![model(), gains(0)], 1).diagnostics.is_empty());
    let matrix = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    assert!(!call(vec![model(), matrix], 1).diagnostics.is_empty());
    assert!(call(vec![model(), gains(2), model()], 0)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RLOCUS-PLOT-MODEL"));
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
        &RLOCUS_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![model(), model(), gains(2)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::CurrentFunctionNargout),
        },
    );
    assert!(!result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RLOCUS-ARITY"));
}
