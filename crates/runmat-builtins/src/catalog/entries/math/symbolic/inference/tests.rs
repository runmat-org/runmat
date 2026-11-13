use super::*;
fn numeric() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
fn symbolic() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Symbolic)
}
fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext {
            literal_args: literals,
        },
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}
fn evaluate(operation: SymbolicScalarOperation, request: &CallRequest) -> CallInference {
    crate::infer_catalog_call(operation.catalog_entry(), request)
}
#[test]
fn all_five_declarations_produce_actual_scalar_classes_and_bounded_outputs() {
    for operation in [
        SymbolicScalarOperation::Precision,
        SymbolicScalarOperation::Integrate,
        SymbolicScalarOperation::Limit,
        SymbolicScalarOperation::Piecewise,
        SymbolicScalarOperation::Decimal,
    ] {
        let args = match operation {
            SymbolicScalarOperation::Precision => vec![],
            SymbolicScalarOperation::Piecewise => vec![symbolic(), numeric()],
            _ => vec![symbolic()],
        };
        let mut call = request(args, vec![]);
        let inferred = evaluate(operation, &call);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs.len(), 1);
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(
            inferred.outputs[0].kind,
            if operation == SymbolicScalarOperation::Precision {
                numeric().kind
            } else {
                ValueKindFact::Symbolic
            }
        );
        assert_eq!(operation.catalog_entry().bindings.len(), 1);
        call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!evaluate(operation, &call).diagnostics.is_empty());
    }
}
#[test]
fn scalar_representation_roles_reject_arrays_complex_residency_and_wrong_controls() {
    for operation in [
        SymbolicScalarOperation::Integrate,
        SymbolicScalarOperation::Limit,
        SymbolicScalarOperation::Piecewise,
        SymbolicScalarOperation::Decimal,
    ] {
        for bad in [
            ValueFact::proven(
                ValueKindFact::Symbolic,
                ShapeFact::Scalar,
                StorageFact::Dense,
            ),
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::Scalar,
                StorageFact::Dense,
            ),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            })),
            ValueFact::proven(
                numeric().kind,
                ShapeFact::from(vec![Some(0), Some(1)]),
                StorageFact::Dense,
            ),
        ] {
            let args = if operation == SymbolicScalarOperation::Piecewise {
                vec![bad, numeric()]
            } else {
                vec![bad]
            };
            assert!(!evaluate(operation, &request(args, vec![]))
                .diagnostics
                .is_empty());
        }
    }
    assert!(!evaluate(
        SymbolicScalarOperation::Integrate,
        &request(vec![symbolic(), numeric()], vec![])
    )
    .diagnostics
    .is_empty());
    assert!(!evaluate(
        SymbolicScalarOperation::Limit,
        &request(vec![symbolic(), numeric(), numeric()], vec![])
    )
    .diagnostics
    .is_empty());
    let mut direction = request(vec![symbolic(), symbolic(), numeric(), numeric()], vec![]);
    assert!(!evaluate(SymbolicScalarOperation::Limit, &direction)
        .diagnostics
        .is_empty());
    direction.arguments[3] = ValueFact::scalar(ValueKindFact::String);
    direction.literals.literal_args = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Number(0.),
        LiteralValue::String(" RIGHT ".into()),
    ];
    assert!(evaluate(SymbolicScalarOperation::Limit, &direction)
        .diagnostics
        .is_empty());
    direction.literals.literal_args[3] = LiteralValue::String("sideways".into());
    assert!(!evaluate(SymbolicScalarOperation::Limit, &direction)
        .diagnostics
        .is_empty());
    let mut resident = numeric();
    resident.residency = ResidencyFact::Device { provider: None };
    assert!(!evaluate(
        SymbolicScalarOperation::Decimal,
        &request(vec![symbolic(), resident], vec![])
    )
    .diagnostics
    .is_empty());
}
#[test]
fn literals_precision_and_effects_match_distinct_forms() {
    let mut digits = request(vec![numeric()], vec![LiteralValue::Number(2.5)]);
    assert!(evaluate(SymbolicScalarOperation::Precision, &digits)
        .diagnostics
        .is_empty());
    digits.literals.literal_args[0] = LiteralValue::Number(1.4);
    assert!(!evaluate(SymbolicScalarOperation::Precision, &digits)
        .diagnostics
        .is_empty());
    let read = evaluate(SymbolicScalarOperation::Precision, &request(vec![], vec![]));
    assert!(read.effects.0.contains(&EffectKind::EnvironmentRead));
    assert!(!read.effects.0.contains(&EffectKind::EnvironmentWrite));
    let write = evaluate(
        SymbolicScalarOperation::Precision,
        &request(vec![numeric()], vec![]),
    );
    assert!(write.effects.0.contains(&EffectKind::EnvironmentRead));
    assert!(write.effects.0.contains(&EffectKind::EnvironmentWrite));
    assert!(evaluate(
        SymbolicScalarOperation::Decimal,
        &request(vec![symbolic()], vec![])
    )
    .effects
    .0
    .contains(&EffectKind::EnvironmentRead));
    assert!(!evaluate(
        SymbolicScalarOperation::Decimal,
        &request(vec![symbolic(), numeric()], vec![])
    )
    .effects
    .0
    .contains(&EffectKind::EnvironmentRead));
    let wide = LiteralValue::Integer {
        text: u64::MAX.to_string(),
        class: NumericClass::UInt64,
    };
    let mut value = numeric();
    value.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    });
    assert!(evaluate(
        SymbolicScalarOperation::Decimal,
        &request(vec![value.clone()], vec![wide.clone()])
    )
    .diagnostics
    .is_empty());
    assert!(!evaluate(
        SymbolicScalarOperation::Integrate,
        &request(vec![value], vec![wide])
    )
    .diagnostics
    .is_empty());
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let inferred = evaluate(
        SymbolicScalarOperation::Decimal,
        &request(vec![unknown], vec![]),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Symbolic);
    let option = request(
        vec![
            symbolic(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Hold".into()),
            LiteralValue::Bool(true),
        ],
    );
    assert!(evaluate(SymbolicScalarOperation::Integrate, &option)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.INT.INVALID_OPTION"));
}
