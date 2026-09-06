mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "unsetenv",
    builtin_path = "crate::builtins::io::repl_fs::environment::unsetenv"
)]
async fn unsetenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args)
}
