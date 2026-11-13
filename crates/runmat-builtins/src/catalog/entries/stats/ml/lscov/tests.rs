use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(rows: usize, cols: usize, domain: NumericDomain) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn request(a: ValueFact, b: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![a, b],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn multi_rhs_and_complex_coefficients_have_distinct_output_facts() {
    let inferred = inference::infer(
        &request(
            numeric(5, 3, NumericDomain::Complex),
            numeric(5, 2, NumericDomain::Real),
            3,
        ),
        &LSCOV_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(
        inferred.outputs[1].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(
        inferred.outputs[2].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })
    ));
    assert!(matches!(
        inferred.outputs[1].kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
    ));
}

#[test]
fn fourth_output_requires_vector_and_algorithm_is_typed() {
    let mut call = request(
        numeric(5, 3, NumericDomain::Real),
        numeric(5, 2, NumericDomain::Real),
        4,
    );
    let inferred = inference::infer(&call, &LSCOV_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-LSCOV-S"));
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    call.literals
        .literal_args
        .push(LiteralValue::String("invalid".into()));
    let inferred = inference::infer(&call, &LSCOV_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-LSCOV-ALG"));
}

#[test]
fn rank_one_response_uses_tensor_row_vector_geometry() {
    let response = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &request(numeric(5, 2, NumericDomain::Real), response, 1),
        &LSCOV_CATALOG_ENTRY,
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
fn nonvector_response_with_matching_element_count_still_rejects_wrong_rows() {
    let inferred = inference::infer(
        &request(
            numeric(4, 1, NumericDomain::Real),
            numeric(2, 2, NumericDomain::Real),
            1,
        ),
        &LSCOV_CATALOG_ENTRY,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-LSCOV-ROWS"));
    assert!(!matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));
}

#[test]
fn empty_nd_weight_is_admitted_as_identity() {
    let mut call = request(
        numeric(3, 2, NumericDomain::Real),
        numeric(3, 1, NumericDomain::Real),
        1,
    );
    call.arguments.push(ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(2), Some(3)]),
        StorageFact::Dense,
    ));
    let inferred = inference::infer(&call, &LSCOV_CATALOG_ENTRY);
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
fn non_scalar_algorithm_string_is_not_a_proven_call() {
    let mut call = request(
        numeric(3, 2, NumericDomain::Real),
        numeric(3, 1, NumericDomain::Real),
        1,
    );
    call.arguments.push(ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    ));
    let inferred = inference::infer(&call, &LSCOV_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-LSCOV-ALG-SHAPE"));
    assert!(!matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));
}
