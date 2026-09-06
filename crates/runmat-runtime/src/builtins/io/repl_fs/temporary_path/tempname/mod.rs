mod execute;
mod generator;
mod input;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "tempname",
    builtin_path = "crate::builtins::io::repl_fs::temporary_path::tempname"
)]
async fn tempname_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
