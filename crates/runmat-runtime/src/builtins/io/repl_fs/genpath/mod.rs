mod errors;
mod exclusions;
mod execute;
mod input;
mod output;
mod root;
pub(crate) mod specification;
mod traversal;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "genpath",
    builtin_path = "crate::builtins::io::repl_fs::genpath"
)]
async fn genpath_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
