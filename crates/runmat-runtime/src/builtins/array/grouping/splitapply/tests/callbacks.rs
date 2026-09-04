use super::*;
use crate::builtins::table::table_from_columns;

#[test]
fn supplies_multiple_data_variables_to_each_callback() {
    let output = call(
        "plus",
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]),
        vec![
            tensor(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1]),
            tensor(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1]),
        ],
    )
    .unwrap();
    assert_eq!(numeric(output), vec![11.0, 33.0, 22.0, 44.0]);
}

#[test]
fn table_variables_become_callback_arguments_in_table_order() {
    let table = table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]),
            tensor(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1]),
        ],
    )
    .unwrap();
    let output = call(
        "plus",
        table,
        vec![tensor(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1])],
    )
    .unwrap();
    assert_eq!(numeric(output), vec![11.0, 33.0, 22.0, 44.0]);
}

#[test]
fn concatenates_each_requested_callback_output_independently() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let outputs = output_list(
        call(
            "min",
            tensor(vec![4.0, 1.0, 3.0, 2.0], vec![4, 1]),
            vec![tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1])],
        )
        .unwrap(),
    );
    assert_eq!(numeric(outputs[0].clone()), vec![1.0, 2.0]);
    assert_eq!(numeric(outputs[1].clone()), vec![2.0, 2.0]);
}
