//! Session search-path query and replacement.

mod execute;
mod input;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "path",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::path"
)]
async fn path_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    execute::run(args).await
}
