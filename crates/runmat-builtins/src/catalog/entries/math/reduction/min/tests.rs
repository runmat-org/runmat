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
fn known_pairwise_form_rejects_second_output_and_never_proves_reduction() {
    let result = call(
        numeric(NumericClass::Double, ShapeFact::Scalar),
        vec![LiteralValue::Number(2.0)],
        2,
    );
    assert!(matches!(
        result.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
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
