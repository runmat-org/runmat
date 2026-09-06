mod execute;
mod result;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "setenv",
    builtin_path = "crate::builtins::io::repl_fs::environment::setenv"
)]
async fn setenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args)
}
