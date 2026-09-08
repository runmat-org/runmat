use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, CellFact, DimensionFact,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

fn cell(element: ValueFact, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::Shaped {
            dims: dims.iter().copied().map(DimensionFact::Known).collect(),
        },
        StorageFact::Dense,
    )
}

fn builtin(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId(name.into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

fn infer(arguments: Vec<ValueFact>, literals: LiteralContext) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("cellfun").expect("cellfun entry"),
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn callback_result_and_cell_shape_are_preserved() {
    let result = infer(
        vec![
            builtin("plus"),
            cell(numeric(NumericClass::UInt64), &[2, 3]),
            cell(numeric(NumericClass::UInt64), &[2, 3]),
        ],
        LiteralContext::default(),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(result.outputs[0].shape.element_count(), Some(6));
}

#[test]
fn nonuniform_output_is_a_cell_of_callback_results() {
    let literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("UniformOutput".into()),
        LiteralValue::Bool(false),
    ]);
    let result = infer(
        vec![
            builtin("length"),
            cell(ValueFact::scalar(ValueKindFact::Character), &[1, 2]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        literals,
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(result.outputs[0].shape.element_count(), Some(2));
}

#[test]
fn empty_inputs_preserve_shape_and_have_the_runtime_output_class() {
    let result = infer(
        vec![builtin("abs"), cell(numeric(NumericClass::Int32), &[0, 3])],
        LiteralContext::default(),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape.element_count(), Some(0));
    assert_eq!(
        result.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::Double)
    );
}

#[test]
fn shorthand_and_invalid_shapes_are_typed() {
    let shorthand = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            cell(numeric(NumericClass::Double), &[1, 2]),
        ],
        LiteralContext::new(vec![
            LiteralValue::String("isempty".into()),
            LiteralValue::Unknown,
        ]),
    );
    assert_eq!(shorthand.outputs[0].kind, ValueKindFact::Logical);

    let mismatch = infer(
        vec![
            builtin("plus"),
            cell(numeric(NumericClass::Double), &[1, 2]),
            cell(numeric(NumericClass::Double), &[2, 1]),
        ],
        LiteralContext::default(),
    );
    assert!(mismatch
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-CELLFUN-SIZE"));
}
