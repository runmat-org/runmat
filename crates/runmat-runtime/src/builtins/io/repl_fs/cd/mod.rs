//! Current-working-folder query and mutation.

mod execute;
mod input;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "cd",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::cd"
)]
async fn cd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    execute::run(args).await
}
