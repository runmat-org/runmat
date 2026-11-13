use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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

fn infer(
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: RequestedOutputCount,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("discretize").expect("discretize catalog entry"),
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn default_indices_preserve_shape_and_materialize_double_on_host() {
    let output = infer(
        vec![
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(2), Some(3)]),
            ),
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(1), Some(4)]),
            ),
        ],
        Vec::new(),
        RequestedOutputCount::One,
    );
    assert_eq!(output.outputs.len(), 1);
    assert_eq!(
        output.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert!(matches!(
        output.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
    assert!(matches!(
        output.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    ));
}

#[test]
fn replacement_class_and_computed_edge_output_are_typed_separately() {
    let shape = ShapeFact::from(vec![Some(1), Some(4)]);
    let output = infer(
        vec![
            numeric(NumericClass::Double, shape.clone()),
            numeric(NumericClass::UInt8, ShapeFact::Scalar),
            numeric(
                NumericClass::UInt64,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::UInt8,
            },
            LiteralValue::Unknown,
        ],
        RequestedOutputCount::Exactly(2),
    );
    assert_eq!(output.outputs.len(), 2);
    assert_eq!(output.outputs[0].shape, shape);
    assert!(matches!(
        output.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            ..
        })
    ));
    assert_eq!(
        output.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
}

#[test]
fn explicit_edge_two_output_and_unsupported_input_are_diagnostics() {
    let output = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(1), Some(3)]),
            ),
        ],
        Vec::new(),
        RequestedOutputCount::Exactly(2),
    );
    assert!(output
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DISCRETIZE-X"));
    assert!(output
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DISCRETIZE-OUTPUTS"));
}

#[test]
fn scalar_results_are_dense_and_unknown_count_does_not_reject_second_output() {
    let output = infer(
        vec![
            numeric(NumericClass::Double, ShapeFact::Scalar),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        ],
        vec![],
        RequestedOutputCount::Exactly(2),
    );
    assert!(output.diagnostics.is_empty());
    assert_eq!(output.outputs[0].storage, StorageFact::Dense);
    assert_eq!(
        output.outputs[1].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}

#[test]
fn runtime_option_name_does_not_invent_text_output() {
    let output = infer(
        vec![
            numeric(NumericClass::Double, ShapeFact::Scalar),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(1), Some(3)]),
            ),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![],
        RequestedOutputCount::One,
    );
    assert!(matches!(output.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(output.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn known_matrix_edges_are_rejected_but_unknown_extents_stay_conservative() {
    for (shape, rejected) in [
        (ShapeFact::from(vec![Some(2), Some(2)]), true),
        (ShapeFact::from(vec![Some(1), Some(3), Some(2)]), true),
        (ShapeFact::from(vec![Some(1), Some(3), Some(1)]), false),
        (ShapeFact::from(vec![Some(1), None, None]), false),
    ] {
        let output = infer(
            vec![
                numeric(NumericClass::Double, ShapeFact::Scalar),
                numeric(NumericClass::Double, shape),
            ],
            vec![],
            RequestedOutputCount::One,
        );
        assert_eq!(
            output
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-DISCRETIZE-EDGES"),
            rejected
        );
    }
}
