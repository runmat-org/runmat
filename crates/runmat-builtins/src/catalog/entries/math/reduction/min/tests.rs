use super::*;
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue,
    NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn call(input: ValueFact, controls: Vec<LiteralValue>, outputs: usize) -> CallInference {
    let mut arguments = vec![input];
    arguments.extend(controls.iter().map(|literal| match literal {
        LiteralValue::String(_) | LiteralValue::Character(_) | LiteralValue::Keyword(_) => {
            ValueFact::scalar(ValueKindFact::String)
        }
        LiteralValue::Empty | LiteralValue::Unknown => {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        _ => numeric(NumericClass::Double, ShapeFact::Scalar),
    }));
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &MIN_CATALOG_ENTRY,
    )
}

#[test]
fn literal_zero_is_pairwise_operand_but_not_reduction_axis() {
    let shape = ShapeFact::from(vec![Some(2), Some(3)]);
    let pairwise = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Number(0.0)],
        1,
    );
    assert_eq!(pairwise.outputs[0].shape, shape);
    assert!(!pairwise
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-REDUCTION-DIMENSION"));

    let invalid = call(
        numeric(NumericClass::Double, shape),
        vec![LiteralValue::Empty, LiteralValue::Number(0.0)],
        1,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-REDUCTION-DIMENSION"));
}

#[test]
fn descriptor_preserves_reduction_pairwise_output_and_error_contracts() {
    let labels: Vec<_> = MIN_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    for form in [
        "M = min(A)",
        "[M, I] = min(A)",
        "M = min(A, B)",
        "M = min(A, [], dim)",
        "[M, I] = min(A, [], vecdim)",
        "M = min(A, [], \"all\")",
        "[M, I] = min(A, [], \"linear\")",
        "M = min(A, [], nanflag)",
        "M = min(A, [], \"ComparisonMethod\", method)",
        "M = min(A, B, \"ComparisonMethod\", method)",
    ] {
        assert!(labels.contains(&form), "missing {form}");
    }
    assert!(!labels.contains(&"[M, I] = min(A, B)"));
    assert_eq!(MIN_DESCRIPTOR.errors.len(), 4);
    for code in [
        MIN_ERROR_INVALID_ARGUMENT.code,
        MIN_ERROR_INVALID_INPUT.code,
        MIN_ERROR_SIZE_MISMATCH.code,
        MIN_ERROR_INTERNAL.code,
    ] {
        assert!(MIN_DESCRIPTOR.errors.iter().any(|error| error.code == code));
    }
    assert_eq!(MIN_INTEGER_CAPABILITIES.len(), 2);
}

#[test]
fn reduction_proves_exact_value_and_double_index_shape() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let result = call(numeric(NumericClass::UInt64, shape), vec![], 2);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    assert_eq!(result.outputs[1].shape, result.outputs[0].shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn logical_reduction_uses_double_and_scalar_integer_stays_exact() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let result = call(logical, vec![], 1);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let scalar = call(numeric(NumericClass::UInt64, ShapeFact::Scalar), vec![], 2);
    assert_eq!(
        scalar.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        scalar.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn known_empty_placeholder_selects_reduction_and_unknown_rhs_does_not() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let reduction = call(
        numeric(NumericClass::Single, shape.clone()),
        vec![LiteralValue::Empty, LiteralValue::Number(2.0)],
        2,
    );
    assert_eq!(
        reduction.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        reduction.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    let linear = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Empty, LiteralValue::String("linear".into())],
        2,
    );
    assert_eq!(linear.outputs[0].shape, ShapeFact::Scalar);

    let unresolved = call(
        numeric(NumericClass::Single, shape),
        vec![LiteralValue::Unknown],
        2,
    );
    assert!(matches!(
        unresolved.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    assert!(matches!(
        unresolved.outputs[1].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn conflicting_reduction_axes_leave_facts_dynamic() {
    let result = call(
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(3), Some(4)]),
        ),
        vec![
            LiteralValue::Empty,
            LiteralValue::Number(2.0),
            LiteralValue::String("all".into()),
        ],
        1,
    );
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn known_pairwise_form_proves_scalar_class_and_rejects_second_output() {
    let result = call(
        numeric(NumericClass::Double, ShapeFact::Scalar),
        vec![LiteralValue::Number(2.0)],
        2,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn pairwise_same_class_broadcast_is_proven_after_typed_option_admission() {
    let mut request = CallRequest {
        arguments: vec![
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(1), Some(3)]),
            ),
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(2), Some(1)]),
            ),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ComparisonMethod".into()),
            LiteralValue::String("magnitude".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &MIN_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );

    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("omitnan".into()),
        LiteralValue::Unknown,
    ]);
    let invalid = inference::infer(&request, &MIN_CATALOG_ENTRY);
    assert!(matches!(
        invalid.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
}

#[test]
fn pairwise_known_shape_mismatch_is_diagnosed_without_proving_result() {
    let request = CallRequest {
        arguments: vec![
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(2), Some(3)]),
            ),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(4), Some(3)]),
            ),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &MIN_CATALOG_ENTRY);
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-BROADCAST"));
}

#[test]
fn same_class_single_pairwise_preserves_class_with_unknown_or_scalar_shape() {
    for left_shape in [ShapeFact::Unknown, ShapeFact::Scalar] {
        let request = CallRequest {
            arguments: vec![
                numeric(NumericClass::Single, left_shape.clone()),
                numeric(NumericClass::Single, ShapeFact::Scalar),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Real {
                    text: "2".into(),
                    class: NumericClass::Single,
                },
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = inference::infer(&request, &MIN_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        assert_eq!(result.outputs[0].shape, left_shape);
    }
}

#[test]
fn empty_all_and_resident_results_are_conservative() {
    let empty = call(
        numeric(
            NumericClass::UInt64,
            ShapeFact::from(vec![Some(0), Some(2)]),
        ),
        vec![LiteralValue::Empty, LiteralValue::String("all".into())],
        2,
    );
    assert!(matches!(
        empty.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));

    let mut resident = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(3), Some(1)]),
    );
    resident.residency = ResidencyFact::Device { provider: None };
    let result = call(resident, vec![], 2);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[1].residency, ResidencyFact::Unknown);
}
