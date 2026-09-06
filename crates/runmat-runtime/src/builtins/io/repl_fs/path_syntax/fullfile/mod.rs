mod execute;
mod input;
pub(crate) mod specification;
#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "fullfile",
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::fullfile"
)]
async fn fullfile_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
