use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &FILLMISSING_CATALOG_ENTRY,
    )
}

#[test]
fn rejected_complex_and_sparse_arrays_do_not_claim_numeric_outputs() {
    for (domain, storage) in [
        (NumericDomain::Complex, StorageFact::Dense),
        (NumericDomain::Real, StorageFact::Sparse),
    ] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            storage,
        );
        let result = call(vec![input, ValueFact::scalar(ValueKindFact::String)], 2);
        assert!(result
            .outputs
            .iter()
            .all(|output| output.kind == ValueKindFact::Unknown));
    }
}

#[test]
fn known_numeric_input_preserves_class_and_shape_with_logical_mask() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let result = call(
        vec![input.clone(), ValueFact::scalar(ValueKindFact::String)],
        2,
    );
    assert_eq!(result.outputs[0].shape, input.shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[1].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[1].shape, input.shape);
    assert_eq!(result.outputs[1].storage, StorageFact::Dense);
}

#[test]
fn unknown_aggregate_and_excessive_outputs_are_not_claimed_proven() {
    let result = call(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::String),
        ],
        3,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[1].kind, ValueKindFact::Unknown);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn scalar_constant_replacement_does_not_guess_the_original_class() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("constant".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = inference::infer(&request, &FILLMISSING_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape.element_count(), Some(1));
    assert_eq!(result.outputs[1].kind, ValueKindFact::Logical);
}

#[test]
fn scalar_input_mask_has_a_dense_logical_array_carrier() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let inferred = call(vec![input, ValueFact::scalar(ValueKindFact::String)], 2);
    assert_eq!(inferred.outputs[1].storage, StorageFact::Dense);
    assert_eq!(inferred.outputs[1].shape, ShapeFact::Scalar);
    assert_eq!(inferred.outputs[1].kind, ValueKindFact::Logical);
}

#[test]
fn singleton_tensor_keeps_its_class_while_scalar_replacement_can_stay_resident() {
    let mut input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    input.storage = StorageFact::Dense;
    let mut fill = ValueFact::scalar(ValueKindFact::Logical);
    fill.residency = ResidencyFact::Device { provider: None };
    let mut request = CallRequest {
        arguments: vec![input, ValueFact::scalar(ValueKindFact::String), fill],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("constant".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let tensor = inference::infer(&request, &FILLMISSING_CATALOG_ENTRY);
    assert_eq!(
        tensor.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(tensor.outputs[0].residency, ResidencyFact::Host);
    request.arguments[0].storage = StorageFact::Scalar;
    let scalar = inference::infer(&request, &FILLMISSING_CATALOG_ENTRY);
    assert_eq!(scalar.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(scalar.outputs[0].residency, ResidencyFact::Unknown);
}
