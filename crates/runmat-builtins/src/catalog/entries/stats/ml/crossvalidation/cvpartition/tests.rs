use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn constructor_proves_typed_object_only_for_admitted_known_method() {
    let input = request(
        vec![
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            numeric(),
        ],
        vec![
            LiteralValue::Number(6.0),
            LiteralValue::Character("KFold".into()),
            LiteralValue::Number(3.0),
        ],
    );
    let result = inference::infer_constructor(&input, &CVPARTITION_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected typed object");
    };
    assert!(object
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(runmat_types::standard::CVPARTITION)));
    assert!(object.properties.contains_key("NumTestSets"));
    let invalid = request(
        vec![
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            numeric(),
        ],
        vec![
            LiteralValue::Number(6.0),
            LiteralValue::Character("KFold".into()),
            LiteralValue::Number(7.0),
        ],
    );
    let rejected = inference::infer_constructor(&invalid, &CVPARTITION_CATALOG_ENTRY);
    assert!(!rejected.diagnostics.is_empty());
    assert!(matches!(rejected.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn holdout_fraction_and_count_have_distinct_known_admission() {
    let call = |n: f64, p: f64| {
        request(
            vec![
                numeric(),
                ValueFact::scalar(ValueKindFact::Character),
                numeric(),
            ],
            vec![
                LiteralValue::Number(n),
                LiteralValue::Character("Holdout".into()),
                LiteralValue::Number(p),
            ],
        )
    };
    let fraction = inference::infer_constructor(&call(10.0, 0.2), &CVPARTITION_CATALOG_ENTRY);
    assert!(
        fraction.diagnostics.is_empty(),
        "{:?}",
        fraction.diagnostics
    );
    assert!(matches!(fraction.outputs[0].kind, ValueKindFact::Object(_)));
    let logical_count = request(
        vec![
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Number(10.0),
            LiteralValue::Character("Holdout".into()),
            LiteralValue::Bool(true),
        ],
    );
    let logical = inference::infer_constructor(&logical_count, &CVPARTITION_CATALOG_ENTRY);
    assert!(logical.diagnostics.is_empty(), "{:?}", logical.diagnostics);
    for input in [call(10.0, 10.0), call(1.0, 0.5)] {
        let result = inference::infer_constructor(&input, &CVPARTITION_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }
}

#[test]
fn fractional_kfold_control_is_not_proven_by_dimension_tolerance() {
    let call = request(
        vec![
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            numeric(),
        ],
        vec![
            LiteralValue::Number(10.0),
            LiteralValue::Character("KFold".into()),
            LiteralValue::Number(2.0 + 5.0e-13),
        ],
    );
    let result = inference::infer_constructor(&call, &CVPARTITION_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn independent_kfold_lower_bound_and_known_invalid_options_do_not_prove_object() {
    let unknown_n = request(
        vec![
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            numeric(),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("KFold".into()),
            LiteralValue::Number(1.0),
        ],
    );
    let result = inference::infer_constructor(&unknown_n, &CVPARTITION_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));

    let custom_scalar = request(
        vec![
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(numeric().kind),
        ],
        vec![
            LiteralValue::Character("CustomPartition".into()),
            LiteralValue::Number(1.0),
        ],
    );
    let result = inference::infer_constructor(&custom_scalar, &CVPARTITION_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));

    // A dense singleton is a numeric array, unlike a numeric scalar value.
    // CustomPartition accepts that array representation at runtime.
    let custom_singleton_array = request(
        vec![ValueFact::scalar(ValueKindFact::Character), numeric()],
        vec![
            LiteralValue::Character("CustomPartition".into()),
            LiteralValue::Unknown,
        ],
    );
    let result = inference::infer_constructor(&custom_singleton_array, &CVPARTITION_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Object(_)));

    let stratify_two = request(
        vec![
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            numeric(),
            ValueFact::scalar(ValueKindFact::Character),
            numeric(),
        ],
        vec![
            LiteralValue::Number(10.0),
            LiteralValue::Character("KFold".into()),
            LiteralValue::Number(2.0),
            LiteralValue::Character("Stratify".into()),
            LiteralValue::Number(2.0),
        ],
    );
    let result = inference::infer_constructor(&stratify_two, &CVPARTITION_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn mask_access_checks_object_identity_and_index_keyword() {
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::CVPARTITION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let valid = request(
        vec![object.clone(), ValueFact::scalar(ValueKindFact::Character)],
        vec![LiteralValue::Unknown, LiteralValue::Character("all".into())],
    );
    let result = inference::infer_test(&valid, &TEST_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Logical));
    let bad = request(
        vec![object, ValueFact::scalar(ValueKindFact::Character)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("last".into()),
        ],
    );
    let rejected = inference::infer_training(&bad, &TRAINING_CATALOG_ENTRY);
    assert!(!rejected.diagnostics.is_empty());
    assert!(matches!(rejected.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn known_zero_mask_index_is_invalid_without_a_proven_output() {
    let mut constructed = inference::infer_constructor(
        &request(
            vec![
                numeric(),
                ValueFact::scalar(ValueKindFact::Character),
                numeric(),
            ],
            vec![
                LiteralValue::Number(6.0),
                LiteralValue::Character("KFold".into()),
                LiteralValue::Number(2.0),
            ],
        ),
        &CVPARTITION_CATALOG_ENTRY,
    );
    let object = constructed.outputs.remove(0);
    let input = request(
        vec![object, numeric()],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
    );
    let result = inference::infer_test(&input, &TEST_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}
