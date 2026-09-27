use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(super::execute::run(args))
}

#[test]
fn mixed_cell_and_string_paths_follow_semantic_order_in_every_dimension() {
    for shape in [vec![2, 2], vec![2, 2, 2], vec![0, 2, 3]] {
        let count = shape.iter().product();
        let folders = (0..count).map(|n| format!("folder{n}")).collect::<Vec<_>>();
        let files = (0..count).map(|n| format!("file{n}")).collect::<Vec<_>>();
        let cells = CellArray::from_column_major(
            folders
                .iter()
                .map(|s| Value::CharArray(CharArray::new_row(s)))
                .collect(),
            shape.clone(),
        )
        .unwrap();
        let strings = StringArray::new(files.clone(), shape.clone()).unwrap();
        let Value::StringArray(result) = run(vec![
            Value::Cell(cells.clone()),
            Value::StringArray(strings),
        ])
        .unwrap() else {
            panic!("mixed text returns strings");
        };
        assert_eq!(result.shape, shape);
        for (index, value) in result.data.iter().enumerate() {
            assert_eq!(
                value,
                &super::super::lexical::join(&[&folders[index], &files[index]])
            );
        }
        let Value::Cell(result) = run(vec![
            Value::Cell(cells),
            Value::CharArray(CharArray::new_row("leaf")),
        ])
        .unwrap() else {
            panic!("character and cell paths return cells");
        };
        assert_eq!(result.shape, shape);
        for (index, value) in result.iter_column_major().enumerate() {
            assert_eq!(
                String::try_from(value).unwrap(),
                super::super::lexical::join(&[&folders[index], "leaf"])
            );
        }
    }
}

#[test]
fn nd_character_components_reject_instead_of_losing_pages() {
    let value = Value::CharArray(CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap());
    assert!(run(vec![value.clone()]).is_err());
    assert!(run(vec![Value::Cell(
        CellArray::new(vec![value], 1, 1).unwrap()
    )])
    .is_err());
}

#[test]
fn trailing_singleton_character_components_are_rows() {
    let row = Value::CharArray(CharArray::new_with_shape(vec!['a', 'b'], vec![1, 2, 1]).unwrap());
    assert_eq!(
        run(vec![row]).unwrap(),
        Value::CharArray(CharArray::new_row("ab"))
    );
}

#[test]
fn invalid_output_count_precedes_provider_access() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 2])
            .handle,
    );
    let error = run(vec![resident]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:fullfile:TooManyOutputs"));
}

#[test]
fn joins_scalars_and_preserves_trailing_separator() {
    let result = run(vec![
        Value::CharArray(CharArray::new_row("data")),
        Value::CharArray(CharArray::new_row("raw")),
        Value::CharArray(CharArray::new_row(std::path::MAIN_SEPARATOR_STR)),
    ])
    .unwrap();
    let text = String::try_from(&result).unwrap();
    assert!(text.ends_with(std::path::MAIN_SEPARATOR));
    assert!(text.contains("data"));
}

#[test]
fn broadcasts_scalars_over_string_arrays() {
    let names = StringArray::new(vec!["a.m".into(), "b.m".into()], vec![1, 2]).unwrap();
    let Value::StringArray(paths) =
        run(vec![Value::from("src"), Value::StringArray(names)]).unwrap()
    else {
        panic!("string array")
    };
    assert_eq!(paths.shape, vec![1, 2]);
    assert!(paths.data[0].ends_with("a.m"));
}

#[test]
fn broadcasts_scalars_over_cells_and_rejects_shape_mismatch() {
    let cells = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("a")),
            Value::CharArray(CharArray::new_row("b")),
        ],
        2,
        1,
    )
    .unwrap();
    let Value::Cell(paths) = run(vec![
        Value::CharArray(CharArray::new_row("src")),
        Value::Cell(cells),
    ])
    .unwrap() else {
        panic!("cell")
    };
    assert_eq!(paths.shape, vec![2, 1]);
    let left = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
    let right = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
    assert!(run(vec![Value::StringArray(left), Value::StringArray(right)]).is_err());
}
