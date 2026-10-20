mod arguments;
mod errors;
mod execute;
mod operation;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "rmpath",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::rmpath"
)]
async fn rmpath_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
