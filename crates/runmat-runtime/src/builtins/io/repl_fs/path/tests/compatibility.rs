use std::convert::TryFrom;

use runmat_value::{IntegerStorage, Tensor, Value};

use super::support::{call, PathGuard};
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

fn codes() -> Value {
    Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U16("typed-path".encode_utf16().collect()),
            vec![1, 10],
        )
        .expect("numeric character-code row"),
    )
}

#[test]
fn runmat_mode_accepts_typed_numeric_character_codes() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let _guard = PathGuard::new();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    call(vec![codes()]).expect("numeric path");
    assert_eq!(
        String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
        "typed-path"
    );
}

#[test]
fn matlab_mode_rejects_numeric_codes_before_mutation() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let guard = PathGuard::new();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(vec![codes()]).expect_err("numeric extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:PathNumericCharacterCodesExtension")
    );
    assert_eq!(
        String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
        guard.previous
    );
}
