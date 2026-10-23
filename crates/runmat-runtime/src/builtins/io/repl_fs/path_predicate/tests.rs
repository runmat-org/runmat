use runmat_builtins::{ISFILE_ERROR_ARITY, ISFILE_ERROR_PATH};
use runmat_value::{CellArray, CharArray, Value};

use super::execute;

fn run(args: &[Value], predicate: fn(&runmat_filesystem::FsMetadata) -> bool) -> Value {
    futures::executor::block_on(execute::evaluate(
        crate::call::FunctionArgumentView::from_slice(args),
        "isfile",
        &ISFILE_ERROR_ARITY,
        &ISFILE_ERROR_PATH,
        predicate,
    ))
    .expect("path predicate")
}

#[test]
fn scalar_and_cell_inputs_preserve_predicate_semantics_and_shape() {
    let _guard = runmat_filesystem::provider_override_lock();
    let root = tempfile::tempdir().expect("temporary directory");
    let file = root.path().join("sample.txt");
    std::fs::write(&file, b"sample").expect("test file");

    assert_eq!(
        run(
            &[Value::from(file.to_string_lossy().to_string())],
            runmat_filesystem::FsMetadata::is_file
        ),
        Value::Bool(true)
    );
    let cell = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row(&file.to_string_lossy())),
            Value::CharArray(CharArray::new_row(
                &root.path().join("missing.txt").to_string_lossy(),
            )),
        ],
        1,
        2,
    )
    .expect("cell");
    let Value::LogicalArray(result) =
        run(&[Value::Cell(cell)], runmat_filesystem::FsMetadata::is_file)
    else {
        panic!("logical array")
    };
    assert_eq!(result.shape, vec![1, 2]);
    assert_eq!(result.data, vec![1, 0]);
}

#[test]
fn invalid_input_rejects_before_provider_access() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let error = futures::executor::block_on(execute::evaluate(
        crate::call::FunctionArgumentView::from_slice(&[resident]),
        "isfile",
        &ISFILE_ERROR_ARITY,
        &ISFILE_ERROR_PATH,
        runmat_filesystem::FsMetadata::is_file,
    ))
    .expect_err("resident numeric input");
    assert_eq!(error.identifier(), Some("RunMat:isfile:InvalidPath"));
    assert!(!error.message().contains("provider"));
}

#[test]
fn cell_predicates_preserve_semantic_order_including_nd_and_empty_arrays() {
    let _lock = runmat_filesystem::provider_override_lock();
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("sample");
    std::fs::write(&file, b"value").unwrap();
    for shape in [vec![2, 2], vec![2, 2, 2], vec![0, 2, 3]] {
        let count = shape.iter().product();
        let values = (0..count)
            .map(|index| {
                let path = if index % 3 == 0 {
                    file.clone()
                } else {
                    root.path().to_path_buf()
                };
                Value::CharArray(CharArray::new_row(&path.to_string_lossy()))
            })
            .collect();
        let input = Value::Cell(CellArray::from_column_major(values, shape.clone()).unwrap());
        for (predicate, expected) in [
            (
                runmat_filesystem::FsMetadata::is_file
                    as fn(&runmat_filesystem::FsMetadata) -> bool,
                true,
            ),
            (
                runmat_filesystem::FsMetadata::is_dir as fn(&runmat_filesystem::FsMetadata) -> bool,
                false,
            ),
        ] {
            let Value::LogicalArray(output) = run(std::slice::from_ref(&input), predicate) else {
                panic!("logical array");
            };
            assert_eq!(output.shape, shape);
            assert_eq!(
                output.data,
                (0..count)
                    .map(|i| u8::from((i % 3 == 0) == expected))
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn genuine_character_pages_reject_but_trailing_singletons_remain_rows() {
    for (shape, valid) in [(vec![1, 2, 1], true), (vec![1, 1, 2], false)] {
        let value = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], shape).unwrap(),
        );
        for input in [
            value.clone(),
            Value::Cell(CellArray::new(vec![value], 1, 1).unwrap()),
        ] {
            assert_eq!(
                super::input::parse(
                    crate::call::FunctionArgumentView::from_slice(&[input]),
                    "isfile",
                    &ISFILE_ERROR_ARITY,
                    &ISFILE_ERROR_PATH
                )
                .is_ok(),
                valid
            );
        }
    }
}
