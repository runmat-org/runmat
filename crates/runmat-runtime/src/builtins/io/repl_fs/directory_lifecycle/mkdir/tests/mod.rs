mod creation;
mod validation;

use std::sync::MutexGuard;

use super::super::result::DirectoryOutcome;
use runmat_value::Value;

pub(super) fn lock() -> MutexGuard<'static, ()> {
    runmat_filesystem::provider_override_lock()
}

pub(super) fn evaluate(args: Vec<Value>) -> crate::BuiltinResult<DirectoryOutcome> {
    futures::executor::block_on(super::execute::evaluate(args))
}

pub(super) fn invoke(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::mkdir_builtin(args))
}
