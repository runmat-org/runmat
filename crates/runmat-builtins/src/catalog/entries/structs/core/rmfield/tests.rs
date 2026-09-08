use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount,
    ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    infer_catalog_call(
        &RMFIELD_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn structure(fields: &[&str]) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields: fields
            .iter()
            .map(|name| ((*name).to_string(), unknown()))
            .collect(),
        fields_complete: true,
    }))
}

#[test]
fn scalar_structure_retains_its_container_but_not_an_unproven_schema() {
    let result = infer(vec![
        structure(&["keep", "drop"]),
        ValueFact::scalar(ValueKindFact::String),
    ]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("expected structure fact");
    };
    assert!(output.fields.is_empty());
    assert!(!output.fields_complete);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn represented_array_preserves_shape_and_erases_element_schema() {
    let shape = ShapeFact::from(vec![Some(2), Some(3)]);
    let target = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(structure(&["a", "b"])),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape.clone(),
        StorageFact::Dense,
    );
    let result = infer(vec![target, ValueFact::scalar(ValueKindFact::Character)]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, shape);
    let ValueKindFact::Cell(output) = &result.outputs[0].kind else {
        panic!("expected represented structure-array fact");
    };
    assert!(matches!(output.element.kind, ValueKindFact::Struct(_)));
}

#[test]
fn known_invalid_targets_names_and_missing_arguments_are_diagnostics() {
    assert!(!infer(vec![unknown()]).diagnostics.is_empty());
    assert!(!infer(vec![
        ValueFact::scalar(ValueKindFact::Logical),
        ValueFact::scalar(ValueKindFact::String),
    ])
    .diagnostics
    .is_empty());
    assert!(!infer(vec![
        structure(&[]),
        ValueFact::scalar(ValueKindFact::Logical),
    ])
    .diagnostics
    .is_empty());
}

#[test]
fn catalog_declares_variadic_form_as_a_runmat_extension() {
    assert_eq!(RMFIELD_EXTENSIONS, &[RMFIELD_VARIADIC_EXTENSION]);
    assert_eq!(RMFIELD_DESCRIPTOR.signatures.len(), 2);
}
