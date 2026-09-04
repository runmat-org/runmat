use runmat_value::{CellArray, CharArray, ObjectInstance, StringArray, Tensor, Value};

use super::super::execute;
use super::outputs;

#[tokio::test]
async fn text_groups_use_first_appearance() {
    let input = Value::StringArray(
        StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
    );
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![1.0, 2.0, 1.0]);
    assert!(matches!(&values[2], Value::Cell(cell) if cell.rows == 2));
}

#[tokio::test]
async fn empty_text_is_a_level_while_missing_string_is_not() {
    let input = Value::StringArray(
        StringArray::new(vec!["".into(), "<missing>".into(), "".into()], vec![3, 1]).unwrap(),
    );
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    let indices = g.materialize_f64();
    assert_eq!(indices[0], 1.0);
    assert!(indices[1].is_nan());
    assert_eq!(indices[2], 1.0);
    assert!(matches!(&values[2], Value::Cell(cell) if cell.rows == 1));
}

#[tokio::test]
async fn cellstr_groups_use_first_appearance_and_remain_cellstr() {
    let input = Value::Cell(
        CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("west")),
                Value::CharArray(CharArray::new_row("east")),
                Value::CharArray(CharArray::new_row("west")),
            ],
            1,
            3,
        )
        .unwrap(),
    );
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![1.0, 2.0, 1.0]);
    assert!(matches!(&values[2], Value::Cell(cell) if cell.rows == 2 && cell.cols == 1));
}

#[tokio::test]
async fn character_matrix_treats_each_row_as_one_label() {
    let input = Value::CharArray(CharArray::new(vec!['b', ' ', 'a', ' ', 'b', ' '], 3, 2).unwrap());
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![1.0, 2.0, 1.0]);
    assert!(matches!(&values[2], Value::CharArray(chars) if chars.rows == 2 && chars.cols == 2));
}

#[tokio::test]
async fn categorical_order_retains_unobserved_categories() {
    let mut categorical = ObjectInstance::new(runmat_types::standard::CATEGORICAL.to_string());
    categorical.properties.insert(
        "Codes".into(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0, f64::NAN], vec![3, 1]).unwrap()),
    );
    categorical.properties.insert(
        "Categories".into(),
        Value::StringArray(
            StringArray::new(
                vec!["unused".into(), "seen".into(), "later".into()],
                vec![1, 3],
            )
            .unwrap(),
        ),
    );
    categorical
        .properties
        .insert("Ordinal".into(), Value::Bool(false));

    let values = outputs(execute::apply(Value::Object(categorical)).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    let indices = g.materialize_f64();
    assert_eq!(indices[0], 2.0);
    assert_eq!(indices[1], 2.0);
    assert!(indices[2].is_nan());
    let Value::Object(levels) = &values[2] else {
        panic!("expected categorical levels")
    };
    let Value::Tensor(codes) = &levels.properties["Codes"] else {
        panic!("expected codes")
    };
    assert_eq!(codes.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

#[tokio::test]
async fn datetime_groups_use_first_appearance_and_preserve_datetime_levels() {
    let input = crate::builtins::datetime::datetime_object_from_serial_tensor(
        Tensor::new(vec![738_522.0, 738_156.0, 738_522.0], vec![3, 1]).unwrap(),
        "dd-MMM-yyyy",
    )
    .unwrap();
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![1.0, 2.0, 1.0]);
    let Value::Object(levels) = &values[2] else {
        panic!("expected datetime levels")
    };
    assert!(levels.is_class(runmat_types::standard::DATETIME));
    let serials = crate::builtins::datetime::serials_from_datetime_value(&values[2]).unwrap();
    assert_eq!(serials.shape, vec![2, 1]);
    assert_eq!(serials.materialize_f64(), vec![738_522.0, 738_156.0]);
}

#[tokio::test]
async fn duration_groups_use_first_appearance_and_preserve_duration_levels() {
    let input = crate::builtins::duration::duration_object_from_days_tensor(
        Tensor::new(vec![2.0, 1.0, 2.0], vec![3, 1]).unwrap(),
        "hh:mm:ss",
    )
    .unwrap();
    let values = outputs(execute::apply(input).await.unwrap());
    let Value::Tensor(g) = &values[0] else {
        panic!("expected g")
    };
    assert_eq!(g.materialize_f64(), vec![1.0, 2.0, 1.0]);
    let Value::Object(levels) = &values[2] else {
        panic!("expected duration levels")
    };
    assert!(levels.is_class(runmat_types::standard::DURATION));
    let days = crate::builtins::duration::duration_tensor_from_duration_value(&values[2]).unwrap();
    assert_eq!(days.shape, vec![2, 1]);
    assert_eq!(days.materialize_f64(), vec![2.0, 1.0]);
}
