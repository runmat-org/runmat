mod contents;
mod errors;
mod execute;
mod input;
mod persistence;
mod result;
mod target;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "savepath",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::savepath"
)]
async fn savepath_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
