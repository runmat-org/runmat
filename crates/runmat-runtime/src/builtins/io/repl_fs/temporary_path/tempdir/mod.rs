mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "tempdir",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::temporary_path::tempdir"
)]
async fn tempdir_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    execute::run(args)
}
