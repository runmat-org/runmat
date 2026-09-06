use std::collections::BTreeMap;

use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    standard, CallRequest, CellFact, DynamicReason, LiteralContext, NumericClass, NumericDomain,
    NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn infer(
    name: &str,
    arguments: Vec<ValueFact>,
    outputs: RequestedOutputCount,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name(name).expect("environment catalog entry"),
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}

fn shaped(kind: ValueKindFact, dimensions: &[usize]) -> ValueFact {
    ValueFact::proven(
        kind,
        ShapeFact::from(dimensions.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn character_row() -> ValueFact {
    shaped(ValueKindFact::Character, &[1, 8])
}

#[test]
fn getenv_distinguishes_dictionary_scalar_and_container_results() {
    let all = infer("getenv", vec![], RequestedOutputCount::One);
    assert!(matches!(
        &all.outputs[0].kind,
        ValueKindFact::Object(ObjectFact { runtime_class: Some(class), .. })
            if class.is(standard::DICTIONARY)
    ));
    assert_eq!(all.outputs[0].storage, StorageFact::Opaque);

    let scalar_string = infer(
        "getenv",
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::One,
    );
    assert!(matches!(
        scalar_string.outputs[0].kind,
        ValueKindFact::Character
    ));

    let string_array = infer(
        "getenv",
        vec![shaped(ValueKindFact::String, &[2, 3])],
        RequestedOutputCount::One,
    );
    assert_eq!(string_array.outputs[0].kind, ValueKindFact::String);
    assert_eq!(
        string_array.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}

#[test]
fn getenv_maps_known_cell_element_results_without_copying_name_shapes() {
    let input = shaped(
        ValueKindFact::Cell(CellFact {
            element: Box::new(character_row()),
            elements: vec![character_row(), ValueFact::scalar(ValueKindFact::String)],
            elements_complete: true,
        }),
        &[1, 2],
    );
    let result = infer("getenv", vec![input], RequestedOutputCount::One);
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected cell output")
    };
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(cell.elements[0].shape, ShapeFact::from(vec![Some(1), None]));
    assert_eq!(cell.elements[1].kind, ValueKindFact::String);
}

#[test]
fn predicates_preserve_container_shape_and_reject_known_extensions() {
    let strings = shaped(ValueKindFact::String, &[2, 3]);
    let result = infer("isenv", vec![strings], RequestedOutputCount::One);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );

    let matrix = shaped(ValueKindFact::Character, &[2, 8]);
    for name in ["isenv", "unsetenv"] {
        let rejected = infer(name, vec![matrix.clone()], RequestedOutputCount::One);
        assert!(rejected
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.argument == Some(0)));
    }
}

#[test]
fn setenv_models_two_outputs_and_validates_typed_inputs() {
    let result = infer(
        "setenv",
        vec![character_row(), ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::Exactly(2),
    );
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(
        result.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(result.outputs[1].kind, ValueKindFact::Character);
    assert!(result.diagnostics.is_empty());

    let nonscalar_numeric = shaped(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        &[1, 2],
    );
    let rejected = infer(
        "setenv",
        vec![character_row(), nonscalar_numeric],
        RequestedOutputCount::Zero,
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SETENV-VALUE"));
}

#[test]
fn setenv_recognizes_dictionary_identity_without_source_string_matching() {
    let dictionary = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::DICTIONARY.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    let accepted = infer("setenv", vec![dictionary], RequestedOutputCount::Zero);
    assert!(accepted.diagnostics.is_empty());

    let unrelated = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = infer("setenv", vec![unrelated], RequestedOutputCount::Zero);
    assert!(dynamic.diagnostics.is_empty());
}
