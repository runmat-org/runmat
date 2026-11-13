mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isenv",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::environment::isenv"
)]
async fn isenv_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::builtins::common::env::require_read()?;
    execute::run(args)
}
