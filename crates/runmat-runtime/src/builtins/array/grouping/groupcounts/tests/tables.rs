use super::*;
use crate::builtins::table::{table_from_columns, table_height, table_variable_names_from_object};
use runmat_value::{StringArray, Tensor};

#[test]
fn table_form_returns_group_columns_count_and_percent() {
    let table = table_from_columns(
        vec!["G".into(), "X".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
            ),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
        ],
    )
    .unwrap();
    let Value::Object(output) = call(table, vec![Value::from("G")]).unwrap() else {
        panic!("expected output table");
    };
    assert_eq!(table_height(&output).unwrap(), 2);
    assert_eq!(
        table_variable_names_from_object(&output).unwrap(),
        ["G", "GroupCount", "Percent"]
    );
}
