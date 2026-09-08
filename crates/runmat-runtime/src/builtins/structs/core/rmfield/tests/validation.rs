use super::{run, structure};
use runmat_value::{CellArray, CharArray, Value};

#[test]
fn field_name_argument_is_required() {
    let error = run(structure(&[("a", Value::Num(1.0))]), Vec::new()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:rmfield:NotEnoughInputs"));
}

#[test]
fn target_must_be_a_structure_even_when_the_name_collection_is_empty() {
    let names = CellArray::new(Vec::new(), 0, 0).unwrap();
    let error = run(Value::Num(1.0), vec![Value::Cell(names)]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:rmfield:InvalidTarget"));
}

#[test]
fn mixed_cell_target_is_not_a_represented_structure_array() {
    let target = CellArray::new(
        vec![structure(&[("a", Value::Num(1.0))]), Value::Num(2.0)],
        1,
        2,
    )
    .unwrap();
    let error = run(Value::Cell(target), vec![Value::from("a")]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:rmfield:InvalidTarget"));
}

#[test]
fn empty_and_nonscalar_names_have_distinct_errors() {
    let target = structure(&[("a", Value::Num(1.0))]);
    let empty = run(target.clone(), vec![Value::from("")]).unwrap_err();
    assert_eq!(empty.identifier(), Some("RunMat:rmfield:FieldNameEmpty"));

    let matrix = CharArray::new(vec!['a', 'b'], 2, 1).unwrap();
    let invalid = run(target, vec![Value::CharArray(matrix)]).unwrap_err();
    assert_eq!(invalid.identifier(), Some("RunMat:rmfield:FieldNameType"));
}

#[test]
fn invalid_cell_element_reports_its_linear_position() {
    let target = structure(&[("a", Value::Num(1.0))]);
    let names = CellArray::new(vec![Value::from("a"), Value::Num(2.0)], 1, 2).unwrap();
    let error = run(target, vec![Value::Cell(names)]).unwrap_err();
    assert!(error.message().contains("cell element 2"));
}
