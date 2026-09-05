//! Current-working-folder query.

mod execute;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "pwd",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::pwd"
)]
async fn pwd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    execute::run(args)
}
