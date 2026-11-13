mod execute;
mod generator;
mod input;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "tempname",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::temporary_path::tempname"
)]
async fn tempname_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
