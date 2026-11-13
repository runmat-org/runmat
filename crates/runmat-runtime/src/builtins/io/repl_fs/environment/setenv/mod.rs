mod execute;
mod result;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "setenv",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::environment::setenv"
)]
async fn setenv_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::builtins::common::env::require_write()?;
    execute::run(args)
}
