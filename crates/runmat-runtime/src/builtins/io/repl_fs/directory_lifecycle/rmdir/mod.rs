mod execute;
mod options;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "rmdir",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::directory_lifecycle::rmdir"
)]
async fn rmdir_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let request = options::parse(args).await?;
    let outcome = execute::remove(request).await;
    super::result::complete(outcome, "rmdir")
}
