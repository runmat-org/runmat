mod arguments;
mod errors;
mod execute;
mod operation;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "addpath",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::addpath"
)]
async fn addpath_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
