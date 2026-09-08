use super::*;
use runmat_types::{CellFact, DimensionFact, LiteralValue};

#[test]
fn builtin_callback_preserves_class_and_broadcast_shape() {
    let result = infer(
        vec![
            builtin("plus"),
            numeric(NumericClass::UInt64, vec![Some(2), Some(1)]),
            numeric(NumericClass::UInt64, vec![Some(1), Some(3)]),
        ],
        LiteralContext::default(),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)]
        }
    );
}

#[test]
fn relational_callback_produces_logical_for_empty_output() {
    let result = infer(
        vec![
            builtin("gt"),
            numeric(NumericClass::Double, vec![Some(0), Some(3)]),
            numeric(NumericClass::Double, vec![Some(1), Some(3)]),
        ],
        LiteralContext::default(),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[0].shape.element_count(), Some(0));
}

#[test]
fn nonuniform_output_is_a_cell_of_callback_results() {
    let output = ValueFact::scalar(ValueKindFact::Character);
    let callback = callable(
        CallableIdentity::DynamicName(runmat_types::SymbolName("callback".into())),
        Some(output.clone()),
    );
    let literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("UniformOutput".into()),
        LiteralValue::Bool(false),
    ]);
    let result = infer(
        vec![
            callback,
            numeric(NumericClass::Double, vec![Some(1), Some(4)]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        literals,
    );
    let ValueKindFact::Cell(CellFact { element, .. }) = &result.outputs[0].kind else {
        panic!("expected cell output: {:?}", result.outputs[0]);
    };
    assert_eq!(**element, output);
    assert_eq!(result.outputs[0].shape.element_count(), Some(4));
}
