mod contents;
mod errors;
mod execute;
mod input;
mod persistence;
mod result;
pub(crate) mod specification;
mod target;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "savepath",
    builtin_path = "crate::builtins::io::repl_fs::savepath"
)]
async fn savepath_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
