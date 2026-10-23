use super::*;
use runmat_filesystem::{File, MemoryFsProvider};
use runmat_value::{IntegerStorage, Tensor};
use std::sync::Arc;
use tempfile::tempdir;

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(ls_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn rows(value: Value) -> Vec<String> {
    let Value::CharArray(array) = value else {
        panic!("expected character array");
    };
    (0..array.rows)
        .map(|row| {
            runmat_value::decode_utf16_code_units(
                &array.data()[row * array.cols..(row + 1) * array.cols],
            )
            .expect("valid UTF-16 fixture")
            .trim_end()
            .to_string()
        })
        .collect()
}

#[test]
fn runmat_mode_lists_one_name_per_row() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("alpha.txt")).expect("file");
    std::fs::create_dir(folder.path().join("nested")).expect("folder");
    let listed = rows(
        run(vec![Value::from(
            folder.path().to_string_lossy().into_owned(),
        )])
        .expect("ls"),
    );
    assert!(listed.contains(&"alpha.txt".into()));
    assert!(listed.contains(&format!("nested{}", std::path::MAIN_SEPARATOR)));
}

#[test]
fn wildcard_uses_the_filesystem_enumerator() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("one.m")).expect("m file");
    File::create(folder.path().join("two.txt")).expect("text file");
    let pattern = format!("{}/*.m", folder.path().to_string_lossy());
    let listed = rows(run(vec![Value::from(pattern)]).expect("ls wildcard"));
    assert_eq!(listed.len(), 1);
    assert!(listed[0].ends_with("one.m"));
}

#[test]
fn relative_wildcard_output_is_independent_of_provider_path_style() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/one.m", b"")
        .expect("m file");
    filesystem
        .write_project_path("/workspace/two.txt", b"")
        .expect("text file");

    let listed = runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        rows(run(vec![Value::from("*.m")]).expect("provider wildcard"))
    });
    assert_eq!(listed, vec!["one.m"]);
}

#[cfg(not(windows))]
#[test]
fn strict_mode_uses_a_unix_character_vector() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("a.txt")).expect("first file");
    File::create(folder.path().join("b.txt")).expect("second file");
    let Value::CharArray(listing) = run(vec![Value::from(
        folder.path().to_string_lossy().into_owned(),
    )])
    .expect("ls") else {
        panic!("expected character array");
    };
    assert_eq!(listing.rows, 1);
    assert!(runmat_value::decode_utf16_code_units(listing.data())
        .expect("valid UTF-16 fixture")
        .contains("a.txt"));
    assert!(runmat_value::decode_utf16_code_units(listing.data())
        .expect("valid UTF-16 fixture")
        .contains("b.txt"));
}

#[test]
fn invalid_numeric_input_rejects_without_gathering() {
    let tensor = Tensor::new_integer(IntegerStorage::U8(vec![65]), vec![1, 1]).expect("tensor");
    let error = run(vec![Value::Tensor(tensor)]).expect_err("invalid input");
    assert_eq!(error.message(), runmat_builtins::LS_ERROR_NAME.message);
}

#[test]
fn output_and_character_shape_admission_precede_enumeration() {
    let outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert_eq!(
        run(vec![Value::Num(1.0)]).unwrap_err().identifier(),
        runmat_builtins::LS_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    drop(outputs);
    let pages =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap();
    assert_eq!(
        run(vec![Value::CharArray(pages)]).unwrap_err().identifier(),
        runmat_builtins::LS_ERROR_NAME.identifier
    );
}

#[test]
fn console_listing_occurs_only_without_captured_outputs() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/listed.m", b"")
        .unwrap();
    runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        for count in [0, 1] {
            let _requested =
                crate::output_count::push_output_count(Some(count)).expect("output-count scope");
            let _presentation =
                crate::output_context::push_output_count(count).expect("output-count scope");
            let capture = crate::console::begin_capture().expect("console capture");
            run(vec![]).unwrap();
            let text = capture.finish().expect("captured text");
            assert_eq!(text.contains("listed.m"), count == 0);
        }
    });
}
