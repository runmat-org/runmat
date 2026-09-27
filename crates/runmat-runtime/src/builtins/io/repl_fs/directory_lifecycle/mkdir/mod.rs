mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "mkdir",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::directory_lifecycle::mkdir"
)]
async fn mkdir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let outcome = execute::evaluate(args).await?;
    super::result::complete(outcome, "mkdir")
}
