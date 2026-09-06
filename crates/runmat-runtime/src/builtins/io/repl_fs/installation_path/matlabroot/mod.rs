mod execute;
pub(crate) mod specification;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "matlabroot",
    builtin_path = "crate::builtins::io::repl_fs::installation_path::matlabroot"
)]
async fn matlabroot_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    execute::run(args)
}
