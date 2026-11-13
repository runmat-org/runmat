use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    let mut literals = vec![LiteralValue::Unknown; arguments.len()];
    if literals.len() >= 2 {
        literals[1] = LiteralValue::String("previous".into());
    }
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(literals),
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
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let inferred = call(vec![input, ValueFact::scalar(ValueKindFact::String)], 2);
    assert_eq!(inferred.outputs[1].storage, StorageFact::Dense);
    assert_eq!(inferred.outputs[1].shape, ShapeFact::Scalar);
    assert_eq!(inferred.outputs[1].kind, ValueKindFact::Logical);
}

#[test]
fn invalid_known_method_and_string_method_do_not_claim_outputs() {
    let numeric = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let string = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    for (input, method, diagnostic) in [
        (numeric, "bogus", "RM.MISSING.INVALID_ARGUMENT"),
        (string, "linear", "RM.MISSING.UNSUPPORTED_TYPE"),
    ] {
        let inferred = inference::infer(
            &CallRequest {
                arguments: vec![input, ValueFact::scalar(ValueKindFact::String)],
                literals: LiteralContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String(method.into()),
                ]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
            &FILLMISSING_CATALOG_ENTRY,
        );
        assert!(inferred
            .outputs
            .iter()
            .all(|fact| fact.kind == ValueKindFact::Unknown && fact.shape == ShapeFact::Unknown));
        assert!(inferred
            .diagnostics
            .iter()
            .any(|item| item.code == diagnostic));
    }
}

#[test]
fn unknown_numeric_storage_does_not_prove_supported_carrier() {
    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Unknown,
    );
    assert_eq!(
        call(
            vec![input.clone(), ValueFact::scalar(ValueKindFact::String)],
            2
        )
        .outputs[1]
            .kind,
        ValueKindFact::Unknown
    );
    input.storage = StorageFact::Opaque;
    input.residency = ResidencyFact::Device { provider: None };
    let admitted = call(vec![input, ValueFact::scalar(ValueKindFact::String)], 2);
    assert_eq!(admitted.outputs[0].storage, StorageFact::Dense);
    assert_eq!(admitted.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(admitted.outputs[1].storage, StorageFact::Dense);
}

#[test]
fn constant_type_and_dimension_options_control_admitted_facts() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let method = ValueFact::scalar(ValueKindFact::String);
    let text_fill = ValueFact::scalar(ValueKindFact::String);
    let invalid = inference::infer(
        &CallRequest {
            arguments: vec![input.clone(), method.clone(), text_fill],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("constant".into()),
                LiteralValue::String("text".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
        &FILLMISSING_CATALOG_ENTRY,
    );
    assert!(invalid
        .outputs
        .iter()
        .all(|fact| fact.kind == ValueKindFact::Unknown));

    let axis = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    for option_literals in [
        vec![LiteralValue::Vector(vec![LiteralValue::Integer {
            text: "2".into(),
            class: NumericClass::UInt16,
        }])],
        vec![
            LiteralValue::String("dim".into()),
            LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::UInt16,
            },
        ],
    ] {
        let mut arguments = vec![input.clone(), method.clone()];
        if option_literals.len() == 2 {
            arguments.push(ValueFact::scalar(ValueKindFact::String));
        }
        arguments.push(axis.clone());
        let mut literals = vec![LiteralValue::Unknown, LiteralValue::String("linear".into())];
        literals.extend(option_literals);
        let inferred = inference::infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
            &FILLMISSING_CATALOG_ENTRY,
        );
        assert_eq!(
            inferred.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        assert_eq!(inferred.outputs[1].kind, ValueKindFact::Logical);
    }
}

#[test]
fn row_character_method_literal_is_admitted_like_scalar_string() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let method = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(6)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![input.clone(), method],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Character("linear".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
        &FILLMISSING_CATALOG_ENTRY,
    );
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[0].shape, input.shape);
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
    let mut request = CallRequest {
        arguments: vec![
            input,
            ValueFact::scalar(ValueKindFact::String),
            fill.clone(),
        ],
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
    fill.residency = ResidencyFact::Device { provider: None };
    request.arguments[2] = fill;
    let scalar = inference::infer(&request, &FILLMISSING_CATALOG_ENTRY);
    assert_eq!(scalar.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(scalar.outputs[0].residency, ResidencyFact::Unknown);
}
