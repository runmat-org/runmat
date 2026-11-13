use super::*;

fn call(
    class: NumericClass,
    domain: NumericDomain,
    operation: MovingOperation,
    dims: &[usize],
    controls: Vec<LiteralValue>,
    outputs: usize,
) -> CallInference {
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    );
    let mut arguments = vec![input];
    arguments.resize(
        literals.len(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &super::super::movsum::MOVSUM_CATALOG_ENTRY,
        operation,
    )
}

#[test]
fn native_and_promoted_classes_follow_typed_operation_not_entry_spelling() {
    for (operation, input, domain, expected) in [
        (
            MovingOperation::Min,
            NumericClass::UInt64,
            NumericDomain::Real,
            NumericClass::UInt64,
        ),
        (
            MovingOperation::Sum,
            NumericClass::UInt64,
            NumericDomain::Real,
            NumericClass::Double,
        ),
        (
            MovingOperation::Mean,
            NumericClass::Single,
            NumericDomain::Real,
            NumericClass::Single,
        ),
        (
            MovingOperation::Median,
            NumericClass::Single,
            NumericDomain::Complex,
            NumericClass::Single,
        ),
        (
            MovingOperation::Var,
            NumericClass::Single,
            NumericDomain::Complex,
            NumericClass::Single,
        ),
    ] {
        let result = call(
            input,
            domain,
            operation,
            &[1, 4],
            vec![LiteralValue::Number(3.0)],
            1,
        );
        assert_eq!(
            result.outputs[0].numeric(),
            Some(NumericFact {
                class: expected,
                domain: if operation.dispersion() {
                    NumericDomain::Real
                } else {
                    domain
                },
            })
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(4)])
        );
    }
    let integer_dispersion = call(
        NumericClass::UInt64,
        NumericDomain::Real,
        MovingOperation::Std,
        &[1, 4],
        vec![LiteralValue::Number(3.0)],
        1,
    );
    assert_eq!(integer_dispersion.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn empty_geometry_discard_dimension_and_dynamic_controls_are_not_lost() {
    for dims in [
        vec![Some(2), Some(2)],
        vec![Some(1), Some(1), Some(4)],
        vec![Some(1), Some(4), Some(1)],
        vec![None, Some(4)],
    ] {
        let invalid = dims == [Some(2), Some(2)] || dims == [Some(1), Some(1), Some(4)];
        let fact = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(dims),
            StorageFact::Dense,
        );
        for window in [true, false] {
            let mut request = CallRequest {
                arguments: vec![
                    fact.clone(),
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                ],
                literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            };
            if window {
                request.arguments[1] = fact.clone();
            } else {
                request.arguments.extend([
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                    fact.clone(),
                ]);
                request.literals.literal_args.extend([
                    LiteralValue::String("SamplePoints".into()),
                    LiteralValue::Unknown,
                ]);
            }
            let mut diagnostics = Vec::new();
            controls::validate_shapes(&request, &mut diagnostics);
            assert_eq!(!diagnostics.is_empty(), invalid);
        }
    }
    let empty = call(
        NumericClass::Double,
        NumericDomain::Real,
        MovingOperation::Sum,
        &[2, 0],
        vec![LiteralValue::Number(3.0)],
        1,
    );
    assert_eq!(
        empty.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(0)])
    );
    let discard = call(
        NumericClass::Double,
        NumericDomain::Real,
        MovingOperation::Max,
        &[1, 4],
        vec![
            LiteralValue::Number(3.0),
            LiteralValue::Character("Endpoints".into()),
            LiteralValue::Character("discard".into()),
        ],
        1,
    );
    assert_eq!(
        discard.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    let dynamic = call(
        NumericClass::Double,
        NumericDomain::Real,
        MovingOperation::Sum,
        &[1, 4],
        vec![LiteralValue::Unknown],
        2,
    );
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
    assert!(dynamic
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
