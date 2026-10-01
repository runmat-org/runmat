use super::*;
use runmat_types::{
    CallRequest, ClassIdentity, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn numeric(rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn object(class: Option<ClassIdentity>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: class,
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn tabular_identity_is_known_only_for_table_and_timetable() {
    for class in [
        Some(runmat_types::standard::TABLE.owned()),
        Some(runmat_types::standard::TIMETABLE.owned()),
        None,
    ] {
        let call = request(
            vec![object(class), numeric(4, 1)],
            vec![LiteralValue::Unknown; 2],
            1,
        );
        assert!(inference::infer(&call, &FITCTREE_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
    let other = request(
        vec![object(Some(
            runmat_types::standard::CLASSIFICATION_TREE.owned(),
        ))],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(inference::infer(&other, &FITCTREE_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITCTREE-DATA"));
}

#[test]
fn matrix_fit_returns_a_typed_host_tree() {
    let call = request(
        vec![numeric(4, 2), numeric(4, 1)],
        vec![LiteralValue::Unknown; 2],
        1,
    );
    let result = inference::infer(&call, &FITCTREE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(model) = &result.outputs[0].kind else {
        panic!("fitctree must infer a model object");
    };
    assert_eq!(
        model.runtime_class,
        Some(runmat_types::standard::CLASSIFICATION_TREE.owned())
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Opaque);
}

#[test]
fn unknown_data_has_a_model_success_fact() {
    let call = request(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        1,
    );
    let result = inference::infer(&call, &FITCTREE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Object(_)));
}

#[test]
fn known_unsupported_option_and_nonnumeric_input_are_diagnosed() {
    let bad = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("bad".into())],
        1,
    );
    assert!(!inference::infer(&bad, &FITCTREE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let bad_option = request(
        vec![
            numeric(4, 1),
            numeric(4, 1),
            ValueFact::scalar(ValueKindFact::String),
            numeric(1, 1),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Prune".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(!inference::infer(&bad_option, &FITCTREE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn requested_output_count_is_fixed_at_one() {
    let args = vec![numeric(4, 1), numeric(4, 1)];
    let none = request(args.clone(), vec![LiteralValue::Unknown; 2], 0);
    assert!(inference::infer(&none, &FITCTREE_CATALOG_ENTRY)
        .outputs
        .is_empty());
    let two = request(args, vec![LiteralValue::Unknown; 2], 2);
    assert!(!inference::infer(&two, &FITCTREE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
