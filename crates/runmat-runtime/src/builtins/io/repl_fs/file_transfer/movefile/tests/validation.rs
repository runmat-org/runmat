use super::*;
use runmat_builtins::{
    MOVEFILE_ERROR_FLAG_ARG, MOVEFILE_RESULT_DEST_MISSING, MOVEFILE_RESULT_DEST_NOT_DIR,
    MOVEFILE_RESULT_EMPTY_DEST, MOVEFILE_RESULT_EMPTY_SOURCE, MOVEFILE_RESULT_PATTERN_ERROR,
    MOVEFILE_RESULT_SOURCE_NOT_FOUND,
};
use runmat_value::CharArray;
use std::fs::File;
use tempfile::tempdir;

#[test]
fn reports_missing_and_empty_paths() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let missing = temp.path().join("missing.txt");
    let target = temp.path().join("target.txt");
    let outcome = evaluate(&[text(&missing), text(&target)]).unwrap();
    assert_eq!(
        outcome.message_id(),
        MOVEFILE_RESULT_SOURCE_NOT_FOUND.identifier.unwrap()
    );
    assert!(outcome.message().contains("does not exist"));
    assert_eq!(
        evaluate(&[Value::from(""), text(&target)])
            .unwrap()
            .message_id(),
        MOVEFILE_RESULT_EMPTY_SOURCE.identifier.unwrap()
    );
    File::create(&target).unwrap();
    assert_eq!(
        evaluate(&[text(&target), Value::from("")])
            .unwrap()
            .message_id(),
        MOVEFILE_RESULT_EMPTY_DEST.identifier.unwrap()
    );
}

#[test]
fn validates_force_flag_and_accepts_uppercase_character_row() {
    let error = evaluate(&[Value::from("a"), Value::from("b"), Value::Num(1.0)]).unwrap_err();
    assert_eq!(error.message(), MOVEFILE_ERROR_FLAG_ARG.message);
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source.txt");
    let target = temp.path().join("target.txt");
    File::create(&source).unwrap();
    File::create(&target).unwrap();
    let outcome = evaluate(&[
        text(&source),
        text(&target),
        Value::CharArray(CharArray::new_row("F")),
    ])
    .unwrap();
    assert_eq!(outcome.status(), 1.0);
}

#[test]
fn wildcard_requires_existing_directory_and_valid_pattern() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source.log");
    File::create(&source).unwrap();
    let pattern = temp.path().join("*.log");
    let missing = evaluate(&[text(&pattern), text(&temp.path().join("missing"))]).unwrap();
    assert_eq!(
        missing.message_id(),
        MOVEFILE_RESULT_DEST_MISSING.identifier.unwrap()
    );
    let target_file = temp.path().join("target.log");
    File::create(&target_file).unwrap();
    let not_directory = evaluate(&[text(&pattern), text(&target_file)]).unwrap();
    assert_eq!(
        not_directory.message_id(),
        MOVEFILE_RESULT_DEST_NOT_DIR.identifier.unwrap()
    );
    let invalid = evaluate(&[Value::from("[*.txt"), Value::from("dest")]).unwrap();
    assert_eq!(
        invalid.message_id(),
        MOVEFILE_RESULT_PATTERN_ERROR.identifier.unwrap()
    );
}

#[test]
fn resident_paths_reject_without_provider_access_and_character_pages_do_not_flatten() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let pages = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
    );
    for source in [resident, pages] {
        assert_eq!(
            evaluate(&[source, Value::from("unused")])
                .unwrap_err()
                .identifier(),
            runmat_builtins::MOVEFILE_ERROR_SOURCE_ARG.identifier
        );
    }
}

#[test]
fn no_output_failure_throws_and_success_still_moves() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source");
    let target = temp.path().join("target");
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let run = |args| futures::executor::block_on(super::super::movefile_builtin(None, args));
    assert_eq!(
        run(crate::builtins::common::test_support::builtin_arguments(
            vec![text(&source), text(&target)]
        ))
        .unwrap_err()
        .identifier(),
        MOVEFILE_RESULT_SOURCE_NOT_FOUND.identifier
    );
    File::create(&source).unwrap();
    assert_eq!(
        run(crate::builtins::common::test_support::builtin_arguments(
            vec![text(&source), text(&target)]
        ))
        .unwrap(),
        Value::output_list(vec![])
    );
    assert!(!source.exists() && target.exists());
}
