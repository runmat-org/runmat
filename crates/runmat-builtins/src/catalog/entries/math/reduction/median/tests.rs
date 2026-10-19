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
        _ => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    }));
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &MEDIAN_CATALOG_ENTRY,
    )
}

#[test]
fn descriptor_preserves_forms_and_error_identity() {
    let labels: Vec<_> = MEDIAN_DESCRIPTOR
        .signatures
        .iter()
        .map(|item| item.label)
        .collect();
    for form in [
        "M = median(A)",
        "M = median(A, dim)",
        "M = median(A, vecdim)",
        "M = median(A, \"all\")",
        "M = median(A, missingflag)",
        "M = median(A, axes, missingflag)",
        "M = median(A, missingflag, axes)",
        "M = median(___, Weights=W)",
    ] {
        assert!(labels.contains(&form), "missing {form}");
    }
    let missing = MEDIAN_DESCRIPTOR
        .signatures
        .iter()
        .flat_map(|item| item.inputs)
        .find(|input| input.name == "missingflag")
        .expect("missing flag");
    assert_eq!(missing.default, Some("\"includemissing\""));
    assert_eq!(MEDIAN_DESCRIPTOR.errors.len(), 3);
    for code in [
        MEDIAN_ERROR_INVALID_ARGUMENT.code,
        MEDIAN_ERROR_INVALID_INPUT.code,
        MEDIAN_ERROR_INTERNAL.code,
    ] {
        assert!(MEDIAN_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == code));
    }
}

#[test]
fn literal_axes_and_exact_classes_are_proven() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let along_second = call(
        numeric(NumericClass::UInt64, shape.clone()),
        vec![
            LiteralValue::Number(2.0),
            LiteralValue::Character("omitmissing".into()),
        ],
        1,
    );
    assert_eq!(
        along_second.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        along_second.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    let all = call(
        numeric(NumericClass::Single, shape),
        vec![LiteralValue::String(" all ".into())],
        1,
    );
    assert_eq!(all.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        all.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn logical_class_is_retained_and_empty_class_is_dynamic() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let result = call(logical, vec![], 1);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );

    let empty = call(
        numeric(
            NumericClass::UInt64,
            ShapeFact::from(vec![Some(0), Some(2)]),
        ),
        vec![],
        1,
    );
    assert!(matches!(
        empty.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));
}

#[test]
fn weights_conflicts_and_resident_shape_do_not_overclaim() {
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    for controls in [
        vec![
            LiteralValue::String("Weights".into()),
            LiteralValue::Unknown,
        ],
        vec![
            LiteralValue::String("all".into()),
            LiteralValue::Number(2.0),
        ],
    ] {
        let result = call(numeric(NumericClass::Double, shape.clone()), controls, 1);
        assert!(matches!(
            result.outputs[0].certainty,
            CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        ));
    }
    let mut resident = numeric(NumericClass::Single, shape);
    resident.residency = ResidencyFact::Device { provider: None };
    let result = call(resident, vec![], 2);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
