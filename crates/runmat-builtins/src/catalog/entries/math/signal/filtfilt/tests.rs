use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn infer_with_outputs(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    let literal_count = arguments.len();
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(vec![LiteralValue::Unknown; literal_count]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &FILTFILT_CATALOG_ENTRY,
    )
}

fn infer(arguments: Vec<ValueFact>) -> CallInference {
    infer_with_outputs(arguments, 1)
}

#[test]
fn any_single_input_controls_output_class_and_signal_shape() {
    let double = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let single = numeric(NumericClass::Single, NumericDomain::Real, ShapeFact::Scalar);
    let x = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(40)]),
    );
    let result = infer(vec![single, double, x.clone()]);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, x.shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn known_nonvector_or_invalid_class_is_not_proven() {
    let coefficient = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let matrix = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(3), Some(4)]),
    );
    let result = infer(vec![coefficient.clone(), coefficient.clone(), matrix]);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    let mut sparse = coefficient.clone();
    sparse.storage = StorageFact::Sparse;
    let result = infer(vec![sparse, coefficient.clone(), coefficient]);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn request_count_and_complex_singleton_storage_remain_sound() {
    let real = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let complex = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        ShapeFact::Scalar,
    );
    let arguments = vec![real.clone(), real, complex];
    assert!(infer_with_outputs(arguments.clone(), 0).outputs.is_empty());
    let one = infer_with_outputs(arguments.clone(), 1);
    assert_ne!(one.outputs[0].storage, StorageFact::Scalar);
    for requested in [2, 3] {
        let excess = infer_with_outputs(arguments.clone(), requested);
        assert_eq!(excess.outputs.len(), requested);
        assert!(excess
            .diagnostics
            .iter()
            .any(|diag| diag.code == "RM-TYPE-CALL-ARITY"));
        assert_eq!(excess.outputs[1].kind, ValueKindFact::Unknown);
    }
}
