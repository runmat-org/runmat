use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
fn real_vector(count: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(count)]),
        StorageFact::Dense,
    )
}
fn model() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn threshold_signature_and_visible_examples_are_preserved() {
    let signature = STEPINFO_DESCRIPTOR
        .signatures
        .iter()
        .find(|signature| signature.label == "info = stepinfo(..., 'SettlingTimeThreshold', tol)")
        .expect("threshold name-value form");
    assert_eq!(signature.inputs.len(), 3);
    assert_eq!(signature.outputs.len(), 1);
    for (id, visible, output) in [
        (
            "feedback",
            "fprintf('ans = %g\\n', info.Overshoot);",
            "ans = 0",
        ),
        (
            "exponential-samples",
            "fprintf('ans = %g\\n', info.SteadyStateValue);",
            "ans = 1",
        ),
    ] {
        let example = STEPINFO_CATALOG_ENTRY
            .documentation
            .examples
            .iter()
            .find(|example| example.id == id)
            .expect("legacy scenario");
        assert!(example.program.ends_with(visible), "{id}");
        assert_eq!(example.display_output, Some(output), "{id}");
    }
}
#[test]
fn both_forms_return_typed_scalar_metric_structure() {
    for args in [vec![model()], vec![real_vector(3), real_vector(3)]] {
        let result = crate::infer_catalog_call(&STEPINFO_CATALOG_ENTRY, &request(args, 1));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        let ValueKindFact::Struct(info) = &result.outputs[0].kind else {
            panic!("expected metrics struct")
        };
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert!(info.fields.contains_key("RiseTime"));
        assert!(info.fields.contains_key("SteadyStateValue"));
        assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    }
}
#[test]
fn known_invalid_forms_are_diagnosed_without_rejecting_unknowns() {
    assert!(crate::infer_catalog_call(
        &STEPINFO_CATALOG_ENTRY,
        &request(vec![model(), real_vector(2)], 1)
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-STEPINFO-TIME" || d.code == "RM-CATALOG-STEPINFO-OPTIONS"));
    assert!(crate::infer_catalog_call(
        &STEPINFO_CATALOG_ENTRY,
        &request(vec![real_vector(3), real_vector(2)], 1)
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-STEPINFO-LENGTH"));
    assert!(
        crate::infer_catalog_call(&STEPINFO_CATALOG_ENTRY, &request(vec![model()], 2))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STEPINFO-OUTPUTS")
    );
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        crate::infer_catalog_call(&STEPINFO_CATALOG_ENTRY, &request(vec![unknown], 1))
            .diagnostics
            .is_empty()
    );
}

#[test]
fn known_threshold_option_value_is_checked() {
    let threshold = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut call = request(
        vec![model(), ValueFact::scalar(ValueKindFact::String), threshold],
        1,
    );
    call.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("SettlingTimeThreshold".into()),
        LiteralValue::Number(2.0),
    ]);
    assert!(crate::infer_catalog_call(&STEPINFO_CATALOG_ENTRY, &call)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-STEPINFO-THRESHOLD"));
}
