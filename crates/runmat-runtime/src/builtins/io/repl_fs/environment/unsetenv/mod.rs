mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "unsetenv",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::environment::unsetenv"
)]
async fn unsetenv_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::builtins::common::env::require_write()?;
    execute::run(args)
}
