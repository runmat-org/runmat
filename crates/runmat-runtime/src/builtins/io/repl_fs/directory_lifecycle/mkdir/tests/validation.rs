use std::fs::File;

use runmat_value::Value;
use tempfile::tempdir;

use super::{evaluate, lock};

#[test]
fn rejects_non_text_and_empty_names_before_filesystem_access() {
    let non_text = evaluate(vec![Value::Num(1.0)]).expect_err("non-text path rejects");
    assert_eq!(non_text.identifier(), Some("RunMat:mkdir:InvalidFolder"));

    let empty = evaluate(vec![Value::from("")]).expect_err("empty path rejects");
    assert_eq!(empty.identifier(), Some("RunMat:mkdir:InvalidFolderName"));
}

#[test]
fn rejects_a_rooted_child_in_the_two_input_form() {
    let temp = tempdir().expect("temporary directory");
    let error = evaluate(vec![
        Value::from(temp.path().to_string_lossy().to_string()),
        Value::from(std::path::MAIN_SEPARATOR.to_string()),
    ])
    .expect_err("rooted child rejects");

    assert_eq!(
        error.identifier(),
        Some("RunMat:mkdir:FolderMustBeRelative")
    );
}

#[test]
fn preserves_a_file_at_the_target_and_returns_failure() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("occupied");
    File::create(&target).expect("seed file");

    let outcome = evaluate(vec![Value::from(target.to_string_lossy().to_string())])
        .expect("operational failures are outcomes");

    assert!(!outcome.status());
    assert_eq!(outcome.identifier(), "RunMat:mkdir:TargetNotDirectory");
    assert!(target.is_file());
}

#[test]
fn rejects_character_pages_and_resident_paths_without_gathering() {
    let pages =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap();
    assert_eq!(
        evaluate(vec![Value::CharArray(pages)])
            .unwrap_err()
            .identifier(),
        Some("RunMat:mkdir:InvalidFolder")
    );
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    assert_eq!(
        evaluate(vec![resident]).unwrap_err().identifier(),
        Some("RunMat:mkdir:InvalidFolder")
    );
}

#[test]
fn requested_outputs_preserve_filesystem_effects_and_failure_policy() {
    let _lock = lock();
    let temp = tempdir().unwrap();
    let path = temp.path().join("created");
    let args = vec![Value::from(path.to_string_lossy().to_string())];
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        assert_eq!(
            super::invoke(args.clone()).unwrap(),
            Value::output_list(vec![])
        );
        assert!(path.is_dir());
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
        let Value::OutputList(values) = super::invoke(args).unwrap() else {
            panic!("output list")
        };
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], Value::Bool(true));
        assert_eq!(values[3], Value::Num(0.0));
    }
    let file = temp.path().join("file");
    File::create(&file).unwrap();
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert!(super::invoke(vec![Value::from(file.to_string_lossy().to_string())]).is_err());
    assert!(file.is_file());
}
