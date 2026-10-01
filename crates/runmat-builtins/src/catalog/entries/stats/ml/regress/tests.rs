use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn request(y: ValueFact, x: ValueFact, count: usize) -> CallRequest {
    CallRequest {
        arguments: vec![y, x],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn fitted_and_complete_row_output_facts_are_distinct() {
    let inferred = inference::infer(
        &request(
            numeric(ShapeFact::from(vec![Some(5), Some(1)])),
            numeric(ShapeFact::from(vec![Some(5), Some(2)])),
            5,
        ),
        &REGRESS_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        inferred.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        inferred.outputs[2].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(
        inferred.outputs[3].shape,
        ShapeFact::from(vec![None, Some(2)])
    );
    assert_eq!(
        inferred.outputs[4].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
}

#[test]
fn known_invalid_geometry_and_alpha_do_not_prove_outputs() {
    let mut call = request(
        numeric(ShapeFact::from(vec![Some(4), Some(1)])),
        numeric(ShapeFact::from(vec![Some(3), Some(2)])),
        1,
    );
    let inferred = inference::infer(&call, &REGRESS_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-REGRESS-ROWS"));
    assert!(!matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));
    call.arguments[1] = numeric(ShapeFact::from(vec![Some(4), Some(2)]));
    call.arguments.push(numeric(ShapeFact::Scalar));
    call.literals.literal_args.push(LiteralValue::Number(1.0));
    let inferred = inference::infer(&call, &REGRESS_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-REGRESS-ALPHA-RANGE"));
    assert!(!matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));
}

#[test]
fn logical_alpha_is_not_in_the_numeric_alpha_contract() {
    let mut call = request(
        numeric(ShapeFact::from(vec![Some(3), Some(1)])),
        numeric(ShapeFact::from(vec![Some(3), Some(2)])),
        1,
    );
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::Logical));
    call.literals.literal_args.push(LiteralValue::Bool(false));
    let inferred = inference::infer(&call, &REGRESS_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-REGRESS-ALPHA-CLASS"));
}

#[test]
fn rank_one_response_matches_runtime_row_vector_representation() {
    let inferred = inference::infer(
        &request(
            numeric(ShapeFact::from(vec![Some(3)])),
            numeric(ShapeFact::from(vec![Some(3), Some(2)])),
            1,
        ),
        &REGRESS_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn empty_observations_cannot_prove_a_successful_fit() {
    let inferred = inference::infer(
        &request(
            numeric(ShapeFact::from(vec![Some(0), Some(1)])),
            numeric(ShapeFact::from(vec![Some(0), Some(2)])),
            1,
        ),
        &REGRESS_CATALOG_ENTRY,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-REGRESS-EMPTY"));
    assert!(!matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));
}

#[test]
fn discarded_calls_do_not_require_fit_geometry_but_still_validate_alpha() {
    for (response, design) in [
        (vec![Some(0), Some(1)], vec![Some(0), Some(2)]),
        (vec![Some(2), Some(2)], vec![Some(3), Some(1)]),
    ] {
        let mut call = request(
            numeric(ShapeFact::from(response)),
            numeric(ShapeFact::from(design)),
            0,
        );
        let inferred = inference::infer(&call, &REGRESS_CATALOG_ENTRY);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert!(inferred.outputs.is_empty());

        call.arguments.push(numeric(ShapeFact::Scalar));
        call.literals.literal_args.push(LiteralValue::Number(1.0));
        let invalid = inference::infer(&call, &REGRESS_CATALOG_ENTRY);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|item| { item.code == "RM-CATALOG-REGRESS-ALPHA-RANGE" }));
        assert!(invalid.outputs.is_empty());
    }
}
