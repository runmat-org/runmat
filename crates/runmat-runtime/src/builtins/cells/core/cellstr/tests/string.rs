use super::*;
use runmat_value::StringArray;

#[test]
fn string_array_preserves_shape_and_visible_order() {
    let input = StringArray::new(
        vec!["north".into(), "east".into(), "south".into(), "west".into()],
        vec![2, 2],
    )
    .unwrap();
    let output = cell(call(Value::StringArray(input)).unwrap());
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(strings(&output), vec!["north", "south", "east", "west"]);
    assert_eq!(text(&output.get(1, 0).unwrap()), "east");
}

#[test]
fn n_dimensional_string_array_preserves_shape() {
    let input = StringArray::new(
        (0..8).map(|index| index.to_string()).collect(),
        vec![2, 2, 2],
    )
    .unwrap();
    let output = cell(call(Value::StringArray(input)).unwrap());
    assert_eq!(output.shape, vec![2, 2, 2]);
    assert_eq!(
        output.iter_column_major().map(text).collect::<Vec<_>>(),
        vec!["0", "1", "2", "3", "4", "5", "6", "7"]
    );
}

#[test]
fn empty_string_scalar_and_array_have_distinct_shapes() {
    let scalar = cell(call(Value::String(String::new())).unwrap());
    assert_eq!(scalar.shape, vec![1, 1]);
    let Value::CharArray(text) = &scalar.data[0] else {
        panic!("expected character array");
    };
    assert_eq!(text.shape, vec![1, 0]);

    let array = StringArray::new(Vec::new(), vec![0, 2]).unwrap();
    let empty = cell(call(Value::StringArray(array)).unwrap());
    assert_eq!(empty.shape, vec![0, 2]);
    assert!(empty.data.is_empty());
}
