use super::*;
use runmat_types::{standard, CellFact, ObjectFact};

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
