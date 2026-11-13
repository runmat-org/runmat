use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn tf() -> ValueFact {
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
fn scalar(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}
fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &FEEDBACK_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn tf_and_scalar_forms_return_new_host_tf_object() {
    for args in [
        vec![tf(), tf()],
        vec![tf(), scalar(NumericClass::Double)],
        vec![scalar(NumericClass::Double)],
    ] {
        let result = call(args, vec![], 1);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("tf result")
        };
        assert_eq!(
            object.runtime_class,
            Some(standard::TRANSFER_FUNCTION.owned())
        );
        assert!(object.properties_complete);
        assert_eq!(object.properties.len(), 6);
        assert_eq!(object.properties["Ts"].shape, ShapeFact::Scalar);
        assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    }
}

#[test]
fn invalid_systems_signs_arity_and_outputs_are_diagnosed() {
    let mut ss = tf();
    let ValueKindFact::Object(ref mut object) = ss.kind else {
        unreachable!()
    };
    object.runtime_class = Some(standard::STATE_SPACE.owned());
    let mut unknown_runtime_but_known_other = tf();
    let ValueKindFact::Object(ref mut other) = unknown_runtime_but_known_other.kind else {
        unreachable!()
    };
    other.runtime_class = None;
    other.class = Some(ClassId(7));
    for invalid in [
        ss,
        unknown_runtime_but_known_other,
        ValueFact::proven(
            scalar(NumericClass::Double).kind,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        ),
    ] {
        assert!(call(vec![invalid, tf()], vec![], 1)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEEDBACK-SYSTEM"));
    }
    assert!(call(
        vec![tf(), tf(), scalar(NumericClass::Double)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(0.0)
        ],
        1
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-FEEDBACK-SIGN"));
    assert!(call(vec![], vec![], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-FEEDBACK-ARITY"));
    assert!(call(
        vec![tf(), tf(), scalar(NumericClass::Double), tf()],
        vec![],
        1
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-FEEDBACK-ARITY"));
    assert!(call(vec![tf(), tf()], vec![], 0).outputs.is_empty());
    assert!(!call(vec![tf(), tf()], vec![], 2).diagnostics.is_empty());
}

#[test]
fn unknown_system_remains_admitted_without_inventing_its_input_class() {
    let result = call(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue), tf()],
        vec![],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(FEEDBACK_EXTENSIONS.len(), 8);
    assert_eq!(FEEDBACK_INTEGER_CAPABILITIES.len(), 2);
}
