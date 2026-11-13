use super::{names::EnvironmentNames, values::EnvironmentValues};
use runmat_value::{CellArray, CharArray, StringArray, Value};

fn character(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[test]
fn cell_names_and_values_use_semantic_order_and_keep_all_dimensions() {
    for shape in [vec![2, 2], vec![2, 2, 2], vec![0, 2, 3]] {
        let count = shape.iter().product();
        let text = (0..count).map(|n| format!("entry{n}")).collect::<Vec<_>>();
        let cell = Value::Cell(
            CellArray::from_column_major(
                text.iter().map(|text| character(text)).collect(),
                shape.clone(),
            )
            .unwrap(),
        );
        let strings = Value::StringArray(StringArray::new(text.clone(), shape.clone()).unwrap());
        let names = EnvironmentNames::decode(&cell).unwrap();
        assert_eq!(names.names(), text);
        assert_eq!(names.shape(), Some(shape.clone()));
        assert_eq!(names.text_output(text).unwrap(), cell);
        assert_eq!(
            EnvironmentValues::decode(&cell).unwrap(),
            EnvironmentValues::decode(&strings).unwrap()
        );
        let flags = (0..count).map(|n| n % 3 == 0).collect::<Vec<_>>();
        let Value::LogicalArray(result) = names.logical_output(flags.clone()).unwrap() else {
            panic!("cell names require a shaped logical result");
        };
        assert_eq!(result.shape, shape);
        assert_eq!(
            result.data,
            flags.into_iter().map(u8::from).collect::<Vec<_>>()
        );
    }
}

#[test]
fn nd_characters_are_not_silently_truncated_as_names_or_values() {
    let chars = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
    );
    assert!(EnvironmentNames::decode(&chars).is_err());
    assert!(EnvironmentValues::decode(&chars).is_err());
    let cells = Value::Cell(CellArray::new(vec![chars], 1, 1).unwrap());
    assert!(EnvironmentNames::decode(&cells).is_err());
    assert!(EnvironmentValues::decode(&cells).is_err());
}

#[test]
fn trailing_singleton_dimensions_still_describe_a_character_row() {
    let chars = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1]).unwrap(),
    );
    let names = EnvironmentNames::decode(&chars).unwrap();
    assert_eq!(names.names(), &["ab"]);
    assert!(EnvironmentValues::decode(&chars).is_ok());
}

#[test]
fn zero_output_mutation_still_executes_and_cell_string_pairing_is_positional() {
    let _lock = runmat_filesystem::provider_override_lock();
    let names = (0..8)
        .map(|index| format!("RUNMAT_ENV_ND_PAIR_{index}"))
        .collect::<Vec<_>>();
    let cells = Value::Cell(
        CellArray::from_column_major(
            names.iter().map(|name| character(name)).collect(),
            vec![2, 2, 2],
        )
        .unwrap(),
    );
    let values = (0..8)
        .map(|index| format!("value{index}"))
        .collect::<Vec<_>>();
    let strings = Value::StringArray(StringArray::new(values.clone(), vec![2, 2, 2]).unwrap());
    futures::executor::block_on(crate::call_builtin_async_with_outputs(
        "setenv",
        &[cells.clone(), strings],
        0,
    ))
    .unwrap();
    for (name, value) in names.iter().zip(values) {
        assert_eq!(crate::builtins::common::env::var(name).unwrap(), value);
    }
    futures::executor::block_on(crate::call_builtin_async_with_outputs(
        "unsetenv",
        &[cells],
        0,
    ))
    .unwrap();
    assert!(names
        .iter()
        .all(|name| crate::builtins::common::env::var(name).is_err()));
}
