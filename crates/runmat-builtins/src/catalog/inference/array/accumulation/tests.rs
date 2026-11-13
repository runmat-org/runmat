use runmat_types::{
    CallableFact, CapabilitySet, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use super::*;

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

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallRequest {
    runmat_types::CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn default_sum_is_a_host_double_with_inferred_vector_shape() {
    let request = request(
        vec![
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(4), Some(1)]),
            ),
            numeric(
                NumericClass::UInt16,
                ShapeFact::from(vec![Some(4), Some(1)]),
            ),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
    );
    let inference = infer(
        AccumulationInferenceRule::Indexed,
        &request,
        &crate::ACCUMARRAY_CATALOG_ENTRY,
    );
    assert!(matches!(
        inference.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![None, None])
    );
    assert_eq!(inference.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn literal_size_and_sparse_control_refine_output() {
    let arguments = vec![numeric(NumericClass::Double, ShapeFact::Unknown); 6];
    let literals = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Vector(vec![LiteralValue::Number(4.0), LiteralValue::Number(3.0)]),
        LiteralValue::Empty,
        LiteralValue::Empty,
        LiteralValue::Bool(true),
    ];
    let inference = infer(
        AccumulationInferenceRule::Indexed,
        &request(arguments, literals),
        &crate::ACCUMARRAY_CATALOG_ENTRY,
    );
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(3)])
    );
    assert_eq!(inference.outputs[0].storage, StorageFact::Sparse);
}

#[test]
fn known_callback_output_class_is_preserved() {
    let callback_output = numeric(NumericClass::Int16, ShapeFact::Scalar);
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: vec![callback_output],
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }));
    let arguments = vec![
        numeric(NumericClass::Double, ShapeFact::Unknown),
        numeric(NumericClass::Int16, ShapeFact::Unknown),
        numeric(NumericClass::Double, ShapeFact::Unknown),
        callback,
    ];
    let inference = infer(
        AccumulationInferenceRule::Indexed,
        &request(arguments, vec![LiteralValue::Unknown; 4]),
        &crate::ACCUMARRAY_CATALOG_ENTRY,
    );
    assert!(matches!(
        inference.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            ..
        })
    ));
}
