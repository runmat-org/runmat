use super::*;
use crate::builtins::table::{table_from_columns, table_height, table_variable_names_from_object};
use runmat_value::{IntegerStorage, StringArray, Tensor};

#[test]
fn table_form_returns_exact_identifier_columns() {
    let base = 1_u64 << 53;
    let table = table_from_columns(
        vec!["Name".into(), "Id".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![base + 1, base, base + 1]),
                    vec![3, 1],
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(call(table, Vec::new()).unwrap());
    let Value::Object(ids) = &outputs[1] else {
        panic!("expected identifier table");
    };
    assert_eq!(table_height(ids).unwrap(), 2);
    assert_eq!(
        table_variable_names_from_object(ids).unwrap(),
        ["Name", "Id"]
    );
}
