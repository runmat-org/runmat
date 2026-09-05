use runmat_value::{CharArray, StringArray, Tensor, Value};

use super::support::{call, PathGuard};
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

fn locked() -> (std::sync::MutexGuard<'static, ()>, PathGuard) {
    let lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    (lock, PathGuard::new())
}

#[test]
fn rejects_invalid_text_shapes_and_value_kinds() {
    let (_lock, _guard) = locked();
    let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("strings");
    let characters = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("characters");
    for value in [
        Value::StringArray(strings),
        Value::CharArray(characters),
        Value::Num(1.0),
    ] {
        let error = call(vec![value]).expect_err("invalid path input");
        assert_eq!(error.identifier(), Some("RunMat:path:InvalidInput"));
    }
}

#[test]
fn rejects_excess_arguments_before_mutation() {
    let (_lock, guard) = locked();
    let error = call(vec![Value::from("a"), Value::from("b"), Value::from("c")])
        .expect_err("excess arguments");
    assert_eq!(error.identifier(), Some("RunMat:path:TooManyInputs"));
    let current = crate::builtins::common::path_state::current_path_string();
    assert_eq!(current, guard.previous);
}

#[test]
fn rejects_malformed_numeric_character_code_rows() {
    let (_lock, _guard) = locked();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let values = [
        Tensor::new(vec![65.0, 66.0, 67.0, 68.0], vec![2, 2]).expect("matrix"),
        Tensor::new(vec![65.5], vec![1, 1]).expect("fractional code"),
        Tensor::new(vec![0x11_0000 as f64], vec![1, 1]).expect("invalid code point"),
    ];
    for value in values {
        let error = call(vec![Value::Tensor(value)]).expect_err("invalid numeric text");
        assert_eq!(error.identifier(), Some("RunMat:path:InvalidInput"));
    }
}
