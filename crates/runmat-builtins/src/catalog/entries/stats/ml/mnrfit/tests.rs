use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn request(x: ValueFact, y: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![x, y],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn grouped_counts_prove_double_coefficient_geometry() {
    let result = inference::infer(
        &request(numeric(4, 2), numeric(4, 3), 3),
        &MNRFIT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(matches!(result.outputs[2].kind, ValueKindFact::Struct(_)));
}

#[test]
fn vector_response_category_count_remains_dynamic() {
    let result = inference::infer(
        &request(numeric(4, 2), numeric(4, 1), 1),
        &MNRFIT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
}

#[test]
fn impossible_known_matrix_and_response_shapes_never_prove_success() {
    let mut empty_x = numeric(4, 2);
    empty_x.shape = ShapeFact::from(vec![Some(0), None]);
    let mut component_x = numeric(4, 2);
    component_x.shape = ShapeFact::from(vec![Some(4), Some(2), Some(2)]);
    let cases = [
        (empty_x, numeric(4, 1), "RM-CATALOG-MNRFIT-X-EMPTY"),
        (component_x, numeric(4, 1), "RM-CATALOG-MNRFIT-X-SHAPE"),
        (numeric(4, 1), numeric(2, 2), "RM-CATALOG-MNRFIT-Y-SHAPE"),
        (
            numeric(4, 1),
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Dense,
            ),
            "RM-CATALOG-MNRFIT-Y-SHAPE",
        ),
        (
            numeric(4, 1),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(3), Some(5)]),
                StorageFact::Dense,
            ),
            "RM-CATALOG-MNRFIT-Y-SHAPE",
        ),
    ];
    for (x, y, diagnostic) in cases {
        let result = inference::infer(&request(x, y, 1), &MNRFIT_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|item| item.code == diagnostic));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}

#[test]
fn trailing_singletons_and_unknown_extents_remain_admissible() {
    let mut x = numeric(4, 2);
    x.shape = ShapeFact::from(vec![Some(4), Some(2), Some(1)]);
    let result = inference::infer(&request(x, numeric(4, 3), 1), &MNRFIT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());

    let mut y = numeric(4, 3);
    y.shape = ShapeFact::from(vec![Some(4), Some(3), None]);
    let result = inference::infer(&request(numeric(4, 2), y, 1), &MNRFIT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn invalid_known_predictor_or_option_does_not_prove_success() {
    let mut call = request(ValueFact::scalar(ValueKindFact::String), numeric(4, 1), 1);
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    call.arguments.push(numeric(1, 1));
    call.literals.literal_args.extend([
        LiteralValue::String("unknown".into()),
        LiteralValue::Number(1.0),
    ]);
    let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-MNRFIT-X"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-MNRFIT-OPTION"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn public_options_decode_once_to_typed_identities() {
    assert_eq!(
        MnrfitOption::from_public("ITERATIONLIMIT"),
        Some(MnrfitOption::IterationLimit)
    );
    assert_eq!(MnrfitOption::from_public("bogus"), None);
}
