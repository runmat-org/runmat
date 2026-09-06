use runmat_builtins::{ISFILE_ERROR_ARITY, ISFILE_ERROR_PATH};
use runmat_value::{CellArray, CharArray, Value};

use super::execute;

fn run(args: &[Value], predicate: fn(&runmat_filesystem::FsMetadata) -> bool) -> Value {
    futures::executor::block_on(execute::evaluate(
        args,
        "isfile",
        &ISFILE_ERROR_ARITY,
        &ISFILE_ERROR_PATH,
        predicate,
    ))
    .expect("path predicate")
}

#[test]
fn scalar_and_cell_inputs_preserve_predicate_semantics_and_shape() {
    let _guard = super::super::REPL_FS_TEST_LOCK.lock().expect("test lock");
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
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    });
    let error = futures::executor::block_on(execute::evaluate(
        &[resident],
        "isfile",
        &ISFILE_ERROR_ARITY,
        &ISFILE_ERROR_PATH,
        runmat_filesystem::FsMetadata::is_file,
    ))
    .expect_err("resident numeric input");
    assert_eq!(error.identifier(), Some("RunMat:isfile:InvalidPath"));
    assert!(!error.message().contains("provider"));
}
