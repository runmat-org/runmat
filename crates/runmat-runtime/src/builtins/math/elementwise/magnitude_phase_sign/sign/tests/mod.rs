use super::host::*;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntValue, LogicalArray};

fn sign_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::sign_builtin(value))
}

fn assert_complex_close(got: (f64, f64), want: (f64, f64), tol: f64) {
    if got.0.is_nan() && got.1.is_nan() && want.0.is_nan() && want.1.is_nan() {
        return;
    }
    assert!(
        (got.0 - want.0).abs() <= tol && (got.1 - want.1).abs() <= tol,
        "got {got:?}, expected {want:?}, tol {tol}"
    );
}

mod errors;
mod host;
mod integer;
mod provider;
