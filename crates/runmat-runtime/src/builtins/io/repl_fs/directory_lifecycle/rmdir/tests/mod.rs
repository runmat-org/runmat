mod removal;
mod symbolic_links;
mod validation;

use std::sync::MutexGuard;

use runmat_value::Value;

use super::super::super::REPL_FS_TEST_LOCK;
use super::super::result::DirectoryOutcome;

pub(super) fn lock() -> MutexGuard<'static, ()> {
    REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

pub(super) fn evaluate(args: Vec<Value>) -> crate::BuiltinResult<DirectoryOutcome> {
    futures::executor::block_on(async {
        let request = super::options::parse(args).await?;
        Ok(super::execute::remove(request).await)
    })
}

pub(super) fn invoke(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::rmdir_builtin(args))
}
