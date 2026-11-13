use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn matrix(rows: usize, columns: usize, class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

#[test]
fn matrix_and_model_forms_preserve_class_and_known_width() {
    let a = matrix(2, 2, NumericClass::Single, NumericDomain::Complex);
    let b = matrix(2, 3, NumericClass::Single, NumericDomain::Complex);
    let model = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::STATE_SPACE.owned()),
            properties: BTreeMap::from([("A".into(), a.clone()), ("B".into(), b.clone())]),
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut a_trailing = a.clone();
    a_trailing.shape = ShapeFact::from(vec![Some(2), Some(2), Some(1)]);
    let mut b_trailing = b.clone();
    b_trailing.shape = ShapeFact::from(vec![Some(2), Some(3), Some(1)]);
    for arguments in [vec![a, b], vec![model], vec![a_trailing, b_trailing]] {
        let result = crate::infer_catalog_call(&CTRB_CATALOG_ENTRY, &request(arguments));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(6)])
        );
        assert_eq!(
            result.outputs[0].numeric(),
            Some(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn known_mismatch_diagnosed_without_guessing_unknown_inputs() {
    let a = matrix(2, 2, NumericClass::Double, NumericDomain::Real);
    let b = matrix(3, 1, NumericClass::Double, NumericDomain::Real);
    let result = crate::infer_catalog_call(&CTRB_CATALOG_ENTRY, &request(vec![a, b]));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-CTRB-DIMENSIONS"));
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = crate::infer_catalog_call(&CTRB_CATALOG_ENTRY, &request(vec![unknown]));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);

    let a = matrix(2, 2, NumericClass::Double, NumericDomain::Real);
    let mut b = matrix(2, 1, NumericClass::Double, NumericDomain::Real);
    b.shape = ShapeFact::from(vec![Some(2)]);
    let result = crate::infer_catalog_call(&CTRB_CATALOG_ENTRY, &request(vec![a, b]));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
}
