use runmat_value::{StringArray, Value};

pub(super) fn texts(values: &[&str], shape: &[usize]) -> Value {
    Value::StringArray(
        StringArray::new(
            values.iter().map(|text| (*text).to_owned()).collect(),
            shape.to_vec(),
        )
        .unwrap(),
    )
}

pub(super) fn assert_text(value: Value, expected: &[&str], shape: &[usize]) {
    let Value::StringArray(array) = value else {
        panic!("string result")
    };
    assert_eq!(array.shape(), shape);
    assert_eq!(
        array
            .data()
            .iter()
            .map(|text| text.as_str().expect("text result"))
            .collect::<Vec<_>>(),
        expected
    );
}
