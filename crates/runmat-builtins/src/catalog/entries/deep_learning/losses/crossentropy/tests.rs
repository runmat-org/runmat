use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
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

fn infer(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &CROSSENTROPY_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn predicts_scalar_reduction_and_rejects_known_invalid_inputs() {
    let input = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let result = infer(vec![input.clone(), input.clone()], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        result.outputs[0].numeric().map(|number| number.class),
        Some(NumericClass::Double)
    );

    assert!(!infer(vec![input.clone()], 1).diagnostics.is_empty());
    assert!(!infer(
        vec![input.clone(), ValueFact::scalar(ValueKindFact::String)],
        1
    )
    .diagnostics
    .is_empty());
    assert!(!infer(
        vec![
            input.clone(),
            numeric(
                NumericClass::Single,
                ShapeFact::from(vec![Some(4), Some(1)])
            )
        ],
        1,
    )
    .diagnostics
    .is_empty());
    assert!(!infer(vec![input.clone(), input], 2).diagnostics.is_empty());
}

#[test]
fn unknown_input_remains_dynamic_and_provider_contract_is_optional() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = infer(vec![unknown.clone(), unknown], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        CROSSENTROPY_CATALOG_ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Optional
    );
    assert_eq!(
        CROSSENTROPY_CATALOG_ENTRY.placement.residency,
        BuiltinResidencyPolicy::Dynamic
    );
    assert_eq!(CROSSENTROPY_CATALOG_ENTRY.integer_capabilities.len(), 1);
}

#[test]
fn literal_unreduced_shape_is_known_but_dynamic_reduction_is_not() {
    let data = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let arguments = vec![
        data.clone(),
        data.clone(),
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::String),
    ];
    let request = |value: LiteralValue| CallRequest {
        arguments: arguments.clone(),
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Reduction".into()),
            value,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let known = crate::infer_catalog_call(
        &CROSSENTROPY_CATALOG_ENTRY,
        &request(LiteralValue::String("none".into())),
    );
    assert!(known.diagnostics.is_empty(), "{:?}", known.diagnostics);
    assert_eq!(known.outputs[0].shape, data.shape);
    assert_eq!(
        known.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::Single)
    );

    let dynamic =
        crate::infer_catalog_call(&CROSSENTROPY_CATALOG_ENTRY, &request(LiteralValue::Unknown));
    assert!(dynamic.diagnostics.is_empty(), "{:?}", dynamic.diagnostics);
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn rejects_same_element_count_with_different_geometry_and_sparse_storage() {
    let predictions = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let transposed = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(3), Some(2)]),
    );
    let mismatch = infer(vec![predictions.clone(), transposed], 1);
    assert!(mismatch
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-CROSSENTROPY-SHAPE"));

    let mut sparse = predictions.clone();
    sparse.storage = StorageFact::Sparse;
    assert!(infer(vec![sparse, predictions.clone()], 1)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-CROSSENTROPY-INPUT"));

    let unknown_width = numeric(NumericClass::Single, ShapeFact::from(vec![Some(2), None]));
    assert!(infer(vec![predictions, unknown_width], 1)
        .diagnostics
        .is_empty());
}
