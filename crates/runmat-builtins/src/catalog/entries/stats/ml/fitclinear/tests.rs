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
        assert!(inference::infer(&call, &FITCLINEAR_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
    let other = request(
        vec![object(Some(
            runmat_types::standard::CLASSIFICATION_LINEAR.owned(),
        ))],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(inference::infer(&other, &FITCLINEAR_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITCLINEAR-DATA"));
}

#[test]
fn known_matrix_fit_returns_model_and_fit_info() {
    let call = request(
        vec![numeric(4, 1), numeric(4, 1)],
        vec![LiteralValue::Unknown; 2],
        2,
    );
    let result = inference::infer(&call, &FITCLINEAR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(model) = &result.outputs[0].kind else {
        panic!("fitclinear must infer a model object");
    };
    assert_eq!(
        model.runtime_class,
        Some(runmat_types::standard::CLASSIFICATION_LINEAR.owned())
    );
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Struct(_)));
}

#[test]
fn unknown_data_and_variadic_output_padding_remain_sound() {
    let call = request(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        3,
    );
    let result = inference::infer(&call, &FITCLINEAR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Object(_)));
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Struct(_)));
    assert_eq!(
        result.outputs[2].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[2].shape, ShapeFact::Scalar);
}

#[test]
fn invalid_known_data_and_unsupported_option_are_diagnosed() {
    let bad = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("bad".into())],
        1,
    );
    assert!(!inference::infer(&bad, &FITCLINEAR_CATALOG_ENTRY)
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
            LiteralValue::String("CrossVal".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(!inference::infer(&bad_option, &FITCLINEAR_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn discarded_output_request_yields_no_values() {
    let call = request(
        vec![numeric(4, 1), numeric(4, 1)],
        vec![LiteralValue::Unknown; 2],
        0,
    );
    assert!(inference::infer(&call, &FITCLINEAR_CATALOG_ENTRY)
        .outputs
        .is_empty());
}
