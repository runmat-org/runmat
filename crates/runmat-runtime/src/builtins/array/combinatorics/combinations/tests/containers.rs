use super::*;
use runmat_value::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor};

#[test]
fn cartesian_product_uses_rightmost_fastest_order() {
    let named_inputs = table(
        call(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            vec![Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![1, 3]).unwrap(),
            )],
        )
        .unwrap(),
    );
    assert_eq!(table_height(&named_inputs).unwrap(), 6);
    let variables = table_variables(&named_inputs).unwrap();
    let Value::Tensor(first) = &variables.fields["Var1"] else {
        panic!("expected numeric column")
    };
    assert_eq!(first.materialize_f64(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let Value::StringArray(second) = &variables.fields["Var2"] else {
        panic!("expected string column")
    };
    assert_eq!(second.data, vec!["x", "y", "z", "x", "y", "z"]);
}

#[test]
fn logical_character_and_cell_inputs_keep_their_container_contracts() {
    let chars = table(call(Value::CharArray(CharArray::new_row("ab")), Vec::new()).unwrap());
    let variables = table_variables(&chars).unwrap();
    let Value::StringArray(column) = &variables.fields["Var1"] else {
        panic!("expected string column")
    };
    assert_eq!(column.data, vec!["a", "b"]);

    let logical = table(
        call(
            Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .unwrap(),
    );
    let variables = table_variables(&logical).unwrap();
    let Value::LogicalArray(column) = &variables.fields["Var1"] else {
        panic!("expected logical column")
    };
    assert_eq!(column.data.to_vec(), vec![0, 1]);

    let cells = table(
        call(
            Value::Cell(
                CellArray::new(vec![Value::Num(1.0), Value::String("x".into())], 1, 2).unwrap(),
            ),
            Vec::new(),
        )
        .unwrap(),
    );
    let variables = table_variables(&cells).unwrap();
    let Value::Cell(column) = &variables.fields["Var1"] else {
        panic!("expected cell column")
    };
    assert_eq!(
        column.data,
        vec![Value::Num(1.0), Value::String("x".into())]
    );
}

#[test]
fn variable_names_text_is_data_and_empty_products_keep_every_column() {
    let named_inputs = table(
        call(
            Value::String("VariableNames".into()),
            vec![Value::StringArray(
                StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap(),
            )],
        )
        .unwrap(),
    );
    assert_eq!(table_height(&named_inputs).unwrap(), 2);

    let empty = table(
        call(
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )],
        )
        .unwrap(),
    );
    assert_eq!(table_height(&empty).unwrap(), 0);
    let variables = table_variables(&empty).unwrap();
    assert_eq!(variables.fields.len(), 2);
    assert!(
        matches!(&variables.fields["Var1"], Value::StringArray(value) if value.shape == [0, 1])
    );
    assert!(matches!(&variables.fields["Var2"], Value::Tensor(value) if value.shape == [0, 1]));
}

#[test]
fn unsupported_sequence_representations_are_retained_as_single_cell_values() {
    let complex =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap());
    let output = table(call(complex.clone(), Vec::new()).unwrap());
    assert_eq!(table_height(&output).unwrap(), 1);
    let variables = table_variables(&output).unwrap();
    let Value::Cell(column) = &variables.fields["Var1"] else {
        panic!("expected cell column")
    };
    assert_eq!(column.data, vec![complex]);

    let characters = Value::CharArray(CharArray::new(vec!['a', 'b'], 2, 1).unwrap());
    let output = table(call(characters.clone(), Vec::new()).unwrap());
    assert_eq!(table_height(&output).unwrap(), 1);
    let variables = table_variables(&output).unwrap();
    assert!(matches!(
        &variables.fields["Var1"],
        Value::Cell(column) if column.data == [characters]
    ));
}
