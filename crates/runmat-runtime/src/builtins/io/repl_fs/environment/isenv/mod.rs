mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isenv",
    builtin_path = "crate::builtins::io::repl_fs::environment::isenv"
)]
async fn isenv_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args)
}
