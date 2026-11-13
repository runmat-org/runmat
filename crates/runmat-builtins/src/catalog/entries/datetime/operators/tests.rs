use super::*;
use runmat_types::*;
use std::collections::BTreeMap;
const OPERATIONS: [DatetimeBinaryOperation; 8] = [
    DatetimeBinaryOperation::Eq,
    DatetimeBinaryOperation::Ne,
    DatetimeBinaryOperation::Lt,
    DatetimeBinaryOperation::Le,
    DatetimeBinaryOperation::Gt,
    DatetimeBinaryOperation::Ge,
    DatetimeBinaryOperation::Plus,
    DatetimeBinaryOperation::Minus,
];
fn object(class: StaticClassIdentity, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    )
}
fn number(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
fn call(
    operation: DatetimeBinaryOperation,
    arguments: Vec<ValueFact>,
    count: usize,
) -> CallInference {
    let request = CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    };
    let entry = operation.catalog_entry();
    match operation {
        DatetimeBinaryOperation::Eq => inference::infer_eq(&request, entry),
        DatetimeBinaryOperation::Ne => inference::infer_ne(&request, entry),
        DatetimeBinaryOperation::Lt => inference::infer_lt(&request, entry),
        DatetimeBinaryOperation::Le => inference::infer_le(&request, entry),
        DatetimeBinaryOperation::Gt => inference::infer_gt(&request, entry),
        DatetimeBinaryOperation::Ge => inference::infer_ge(&request, entry),
        DatetimeBinaryOperation::Plus => inference::infer_plus(&request, entry),
        DatetimeBinaryOperation::Minus => inference::infer_minus(&request, entry),
    }
}
#[test]
fn method_inventory_default_contracts_documentation_and_class_declaration_are_joined() {
    let declaration = standard_class_declaration(&standard::DATETIME.owned()).unwrap();
    assert_eq!(ENTRIES.len(), 8);
    for operation in OPERATIONS {
        let entry = operation.catalog_entry();
        assert!(declaration
            .methods
            .iter()
            .any(|method| method.callable.display_name().as_deref() == Some(operation.name())));
        assert_eq!(
            entry.descriptor.completion_policy,
            BuiltinCompletionPolicy::MethodOnly
        );
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.documentation.examples.len(), 2);
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
        let arguments = vec![
            object(standard::DATETIME, ShapeFact::Scalar),
            number(ShapeFact::Scalar),
        ];
        for count in [0, 1, 2] {
            let result = call(operation, arguments.clone(), count);
            assert_eq!(result.diagnostics.is_empty(), count <= 1);
            assert_eq!(result.effects, entry.contract.effect_set());
            if count == 1 {
                assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
            }
        }
        assert!(!call(operation, vec![], 1).diagnostics.is_empty());
    }
}
#[test]
fn operation_specific_rhs_classes_and_actual_double_results_are_proven_without_name_dispatch() {
    let scalar = ShapeFact::Scalar;
    for operation in OPERATIONS {
        let lhs = object(standard::DATETIME, scalar.clone());
        for (rhs_class, allowed, double) in [
            (
                standard::DATETIME,
                operation != DatetimeBinaryOperation::Plus,
                operation.is_comparison() || operation == DatetimeBinaryOperation::Minus,
            ),
            (standard::DURATION, !operation.is_comparison(), false),
            (
                standard::CALENDAR_DURATION,
                !operation.is_comparison(),
                false,
            ),
        ] {
            let result = call(
                operation,
                vec![lhs.clone(), object(rhs_class, scalar.clone())],
                1,
            );
            assert_eq!(result.diagnostics.is_empty(), allowed);
            if allowed {
                if double {
                    assert_eq!(
                        result.outputs[0].numeric().unwrap().class,
                        NumericClass::Double
                    );
                } else {
                    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
                        panic!("datetime output")
                    };
                    assert_eq!(object.runtime_class, Some(standard::DATETIME.owned()));
                    assert_eq!(
                        object.properties[DATETIME_SERIAL_PROPERTY]
                            .numeric()
                            .unwrap()
                            .class,
                        NumericClass::Double
                    );
                }
            }
        }
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let result = call(operation, vec![lhs, unknown], 1);
        if operation == DatetimeBinaryOperation::Minus {
            assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        }
    }
}
#[test]
fn scalar_or_exact_full_geometry_includes_empty_low_rank_and_rejects_general_broadcasting() {
    for operation in OPERATIONS {
        for dims in [
            vec![Some(0), Some(5)],
            vec![Some(2), Some(0), Some(3)],
            vec![Some(3)],
        ] {
            let shape = ShapeFact::from(dims);
            let lhs = object(standard::DATETIME, shape.clone());
            let result = call(operation, vec![lhs.clone(), number(ShapeFact::Scalar)], 1);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].shape, shape);
            let result = call(operation, vec![lhs, number(shape.clone())], 1);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].shape, shape);
        }
        let result = call(
            operation,
            vec![
                object(standard::DATETIME, ShapeFact::from(vec![Some(2), Some(1)])),
                number(ShapeFact::from(vec![Some(1), Some(3)])),
            ],
            1,
        );
        assert!(!result.diagnostics.is_empty());
        let mut resident = number(ShapeFact::Scalar);
        resident.residency = ResidencyFact::Device { provider: None };
        assert!(!call(
            operation,
            vec![object(standard::DATETIME, ShapeFact::Scalar), resident],
            1
        )
        .diagnostics
        .is_empty());
    }
}

#[test]
fn authoritative_payload_shape_sparse_rejection_and_exact_integer_literal_bounds_are_not_erased() {
    for operation in OPERATIONS {
        let shape = ShapeFact::from(vec![Some(2), Some(0), Some(3)]);
        let mut lhs = object(standard::DATETIME, ShapeFact::Unknown);
        let ValueKindFact::Object(object) = &mut lhs.kind else {
            unreachable!()
        };
        object
            .properties
            .insert(DATETIME_SERIAL_PROPERTY.into(), number(shape.clone()));
        let result = call(operation, vec![lhs, number(ShapeFact::Scalar)], 1);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, shape);
        let mut sparse = number(ShapeFact::Scalar);
        sparse.storage = StorageFact::Sparse;
        assert!(!call(
            operation,
            vec![inference::datetime(ShapeFact::Scalar), sparse],
            1
        )
        .diagnostics
        .is_empty());
    }
    let mut integer = number(ShapeFact::Scalar);
    integer.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    });
    let request = CallRequest {
        arguments: vec![inference::datetime(ShapeFact::Scalar), integer],
        literals: LiteralContext {
            literal_args: vec![
                LiteralValue::Unknown,
                LiteralValue::Integer {
                    text: u64::MAX.to_string(),
                    class: NumericClass::UInt64,
                },
            ],
        },
        outputs: OutputSelection::default(),
    };
    assert!(
        !inference::infer_plus(&request, &DATETIME_PLUS_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
