use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn numeric(class: NumericClass, rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn request(
    sample: ValueFact,
    training: ValueFact,
    group: ValueFact,
    outputs: usize,
) -> CallRequest {
    CallRequest {
        arguments: vec![sample, training, group],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 3]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn exact_integer_labels_and_numerical_outputs_have_distinct_facts() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Single, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::UInt64, 4, 1),
            5,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    for output in &result.outputs[1..4] {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    }
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(2), None])
    );
    assert!(matches!(result.outputs[4].kind, ValueKindFact::Cell(_)));
}

#[test]
fn invalid_geometry_does_not_prove_successful_outputs() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 2),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::Double, 4, 1),
            1,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.CLASSIFY.COLUMNS"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn character_group_rows_count_labels_not_characters() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(4), Some(3)]),
                StorageFact::Dense,
            ),
            1,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), None])
    );
}

#[test]
fn unknown_labels_and_excess_outputs_remain_conservative() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            unknown,
            6,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn zero_requested_outputs_take_the_runtime_noop_path() {
    let result = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::Symbolic),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::Double, 4, 1),
            0,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result.outputs.is_empty());
    assert!(result.diagnostics.is_empty());
}

#[test]
fn public_discriminant_spellings_decode_to_one_typed_identity() {
    assert_eq!(
        DiscriminantType::from_public("DIAGLINEAR"),
        Some(DiscriminantType::DiagLinear)
    );
    assert_eq!(
        DiscriminantType::from_public("mahalanobis"),
        Some(DiscriminantType::Mahalanobis)
    );
    assert_eq!(DiscriminantType::from_public("bogus"), None);
}
