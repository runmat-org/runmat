mod removal;
mod symbolic_links;
mod validation;

use std::sync::MutexGuard;

use super::super::result::DirectoryOutcome;
use runmat_value::Value;

pub(super) fn lock() -> MutexGuard<'static, ()> {
    runmat_filesystem::provider_override_lock()
}

pub(super) fn evaluate(args: Vec<Value>) -> crate::BuiltinResult<DirectoryOutcome> {
    futures::executor::block_on(async {
        let request = super::options::parse(
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await?;
        Ok(super::execute::remove(request).await)
    })
}

pub(super) fn invoke(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::rmdir_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}
