use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn model(class: runmat_types::StaticClassIdentity, property: &str, count: usize) -> ValueFact {
    let mut properties = BTreeMap::new();
    properties.insert(
        property.into(),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(count)]),
            StorageFact::Dense,
        ),
    );
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
fn call(arguments: Vec<ValueFact>, count: usize) -> CallInference {
    crate::infer_catalog_call(
        &POLE_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    )
}

#[test]
fn known_static_gain_has_real_empty_host_column() {
    let result = call(
        vec![model(standard::TRANSFER_FUNCTION, "Denominator", 1)],
        1,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(1)])
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
}
#[test]
fn nontrivial_polynomial_length_does_not_prove_root_count_or_complexity() {
    let result = call(
        vec![model(standard::TRANSFER_FUNCTION, "Denominator", 3)],
        1,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}
#[test]
fn invalid_model_and_output_count_are_diagnosed() {
    let invalid = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!call(vec![invalid], 1).diagnostics.is_empty());
    assert!(!call(
        vec![model(standard::TRANSFER_FUNCTION, "Denominator", 1)],
        2
    )
    .diagnostics
    .is_empty());
}
#[test]
fn state_matrix_dimension_proves_pole_column_length() {
    let result = call(vec![model(standard::STATE_SPACE, "A", 1)], 1);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
}

#[test]
fn known_non_model_class_without_runtime_identity_is_rejected() {
    let mut value = model(standard::TRANSFER_FUNCTION, "Denominator", 2);
    let ValueKindFact::Object(object) = &mut value.kind else {
        unreachable!()
    };
    object.class = Some(ClassId(7));
    object.runtime_class = None;
    assert!(!call(vec![value], 1).diagnostics.is_empty());
}
