use super::*;
use runmat_types::*;

#[test]
fn rejects_proven_bad_shapes_complexity_lengths_and_classes() {
    let numeric = |class, domain, shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact { class, domain }),
            shape,
            StorageFact::Dense,
        )
    };
    let shape = |dims: Vec<usize>| ShapeFact::from(dims.into_iter().map(Some).collect::<Vec<_>>());
    let real = |class, dims| numeric(class, NumericDomain::Real, shape(dims));
    for arguments in [
        vec![
            real(NumericClass::Double, vec![2, 2]),
            real(NumericClass::Double, vec![4, 1]),
        ],
        vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Complex,
                shape(vec![2, 1]),
            ),
            real(NumericClass::Double, vec![2, 1]),
        ],
        vec![
            real(NumericClass::Double, vec![2, 1]),
            real(NumericClass::Double, vec![3, 1]),
        ],
        vec![
            real(NumericClass::Int16, vec![2, 1]),
            real(NumericClass::UInt16, vec![2, 1]),
        ],
    ] {
        let request = CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(
            !super::inference::infer(&request, &CONFUSIONMAT_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
}

#[test]
fn unknown_order_keyword_does_not_claim_logical_order() {
    let mut request = request(ValueKindFact::Logical, 2);
    request.arguments.push(ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    ));
    request
        .arguments
        .push(ValueFact::unknown(DynamicReason::RuntimeValue));
    let result = super::inference::infer(&request, &CONFUSIONMAT_CATALOG_ENTRY);
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Unknown));
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Unknown, DimensionFact::Unknown]
        }
    );
}
fn request(kind: ValueKindFact, count: usize) -> CallRequest {
    let value = ValueFact::proven(kind, ShapeFact::Scalar, StorageFact::Scalar);
    CallRequest {
        arguments: vec![value.clone(), value],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}
#[test]
fn exact_integer_order_class_is_retained_and_count_is_double() {
    let result = super::inference::infer(
        &request(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            2,
        ),
        &CONFUSIONMAT_CATALOG_ENTRY,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::UInt64
    );
}
#[test]
fn logical_default_order_is_always_false_then_true() {
    let result = super::inference::infer(
        &request(ValueKindFact::Logical, 2),
        &CONFUSIONMAT_CATALOG_ENTRY,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}
#[test]
fn rejected_output_count_is_diagnostic() {
    assert!(!super::inference::infer(
        &request(ValueKindFact::Logical, 3),
        &CONFUSIONMAT_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
