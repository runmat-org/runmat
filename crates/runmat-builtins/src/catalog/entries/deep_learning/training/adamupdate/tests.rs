use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: Vec<usize>, storage: StorageFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.into_iter().map(Some).collect::<Vec<_>>()),
        storage,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &ADAMUPDATE_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn matching_single_precision_outputs_and_exact_integer_iteration() {
    let data = numeric(NumericClass::Single, vec![2, 3], StorageFact::Dense);
    let empty = numeric(NumericClass::Single, vec![0, 0], StorageFact::Dense);
    let iteration = numeric(NumericClass::UInt64, vec![1, 1], StorageFact::Scalar);
    let result = infer(
        vec![data.clone(), data.clone(), empty.clone(), empty, iteration],
        vec![LiteralValue::Unknown; 5],
        3,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 3);
    assert_eq!(result.outputs[0].kind, data.kind);
    assert_eq!(result.outputs[1].shape, data.shape);
    assert_eq!(result.outputs[2].shape, data.shape);
}

#[test]
fn bad_known_shape_control_and_output_count_diagnose() {
    let data = numeric(NumericClass::Double, vec![2, 3], StorageFact::Dense);
    let wrong = numeric(NumericClass::Double, vec![3, 2], StorageFact::Dense);
    let scalar = numeric(NumericClass::Double, vec![1, 1], StorageFact::Scalar);
    let result = infer(
        vec![data.clone(), wrong, data.clone(), data, scalar.clone()],
        vec![LiteralValue::Unknown; 5],
        4,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ADAM-SHAPE"));
    assert!(!result.diagnostics.is_empty());
    let invalid = infer(
        vec![
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            scalar,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(-1.0),
        ],
        1,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ADAM-CONTROL"));
    assert_eq!(
        ADAMUPDATE_DESCRIPTOR.errors[0].identifier,
        Some("RunMat:deepLearning:InvalidInput")
    );
}
